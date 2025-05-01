from torchquantum import QuantumModule, QuantumDevice
from torchquantum.encoding import Encoder
import torch
import torch.nn as nn
import numpy as np

class OneHotAmplitude(Encoder):
    """
    Encodes 2D data using one-hot amplitude encoding
    """
    def __init__(self, dims):
        super().__init__()

        self._dims = dims

        # Define basis states for each register.
        basisX = torch.eye(self._dims[0])
        basisY = torch.eye(self._dims[1])

        # Get cartesian product of indices
        idx_x, idx_y = torch.arange(basisX.size(0)), torch.arange(basisY.size(0))
        grid_x, grid_y = torch.meshgrid(idx_x, idx_y, indexing='ij')

        basis_states = torch.cat(
            (basisX[grid_x.flatten()], basisY[grid_y.flatten()]), dim=1
        ) 

        # Vectorize this later. States must be given as tensor products.
        self._basis_states = torch.zeros(len(basis_states), 2 ** len(basis_states[0]))
        for i, vec in enumerate(basis_states):
            kron_ = torch.tensor([1])
            for val in vec:
                if val:
                    kron_ = torch.kron(kron_, torch.tensor([0, 1]))
                else:
                    kron_ = torch.kron(kron_, torch.tensor([1, 0]))
            self._basis_states[i] = kron_

    def forward(self, qdev, X):
        if qdev.n_wires != sum(self._dims):
            raise ValueError('Quantum Device does not match image dimensions.')
        
        # Normalize image
        norm = torch.sqrt(torch.square(X).sum(dim=(1, 2)))
        X = X / norm.view(-1, 1, 1)

        # Flatten each image
        X_flat = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

        # Encode image in amplitudes of each basis states
        X_state = torch.sum(self._basis_states.unsqueeze(0) * X_flat.unsqueeze(2), dim=1)
        qdev.set_states(X_state)


class RBS(QuantumModule):
    """ 
    Reconfigurable Beam splitter

    :param theta: Angle in
    :param wires: wires to apply RBS
    """
    def __init__(self, theta = None, wires = None):
        super().__init__()

        # Instantiate theta as a torch tensor
        if isinstance(theta, float):
            self._theta = nn.Parameter(torch.tensor([theta]),)
        elif isinstance(theta, torch.Tensor):
            self._theta = nn.Parameter(theta)
        else:
            self._theta = 2 * np.pi * torch.rand(1)

        self._wires = wires

    def forward(self, qdev: QuantumDevice, wires = None):
        if wires is None and self._wires is None:
            raise ValueError("Please specify wires to place RBS gate.")

        wires = wires if wires is not None else self._wires

        c = torch.cos(self._theta).item()
        s = torch.sin(self._theta).item()

        qdev.qubitunitary(
            wires=wires, 
            params=[[1, 0, 0, 0],
                    [0, c, s, 0],
                    [0, -s, c, 0],
                    [0, 0, 0, 1]]
        )

class Conv2dFilter(QuantumModule):
    """ 
    2D Convolutional filter applied to a register within a convolutional layer.
    Each convolutional filter consists of a series of RBS gates applied between 
    each pair of qubits in the register.

    :param kernel_size: Size of the convolutional filter
    """
    def __init__(self, kernel_size):
        super().__init__()

        self._kernel_size = kernel_size

    def forward(self, qdev, start_wire = None):
        wires1 = [[2 * i + start_wire, 2 * i + 1 + start_wire] for i in range(self._kernel_size // 2)
                    for i in range(self._kernel_size // 2)]
        
        wires2 = [[2 * i + 1 + start_wire, 2 * i + 2 + start_wire] 
                  for i in range((self._kernel_size - 1) // 2)]

        if wires2:
            self._rbs_list = [
                RBS(wires=w) for w in (wires1 + wires2) 
                for column in range(self._kernel_size // 2 + 1)
            ]
        else:
            # For kernel size = 2: Single RBS gate
            self._rbs_list = [RBS(wires=wires1[0])]

        for i, rbs in enumerate(self._rbs_list):
            rbs(qdev)


class Conv2d(QuantumModule):
    """ 
    2D Convolutional Layer. 
    Convolutional unitary is repeatedly applied across the two registers of the QCNN.
    
    :param kernel_size: Size of the convolutional filter
    :param stride: Vertical distance the convolutional filter shifts down the register.
        Note: for a stride < kernel_size, convolutions are applied successively to the 
        same set of qubits and translational invariance is not preserved.
    """
    def __init__(self, dims: tuple, kernel_size: int | tuple[int], stride: int | tuple[int] = None):
        super().__init__()

        self._dims = dims
        self._kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        stride = kernel_size if stride is None else stride
        self._stride = (stride, stride) if isinstance(stride, int) else stride    

        self._filterX = Conv2dFilter(self._kernel_size[0])
        self._filterY = Conv2dFilter(self._kernel_size[1])

        self._start_wires_x = [
            i * self._stride[0] for i in range((dims[0] - self._kernel_size[0] + self._stride[0]) // self._stride[0])
            if i * self._stride[0] < dims[0]
        ]
        self._start_wires_y = [
            i * self._stride[1] + dims[0] for i in range((dims[1] - self._kernel_size[1] + self._stride[1]) // self._stride[1])
            if i * self._stride[1] < sum(dims) - self._kernel_size[1]
        ]
    
    def forward(self, qdev):
        for wire in self._start_wires_x:
            self._filterX(qdev, start_wire=wire)
        
        for wire in self._start_wires_y:
            self._filterY(qdev, start_wire=wire)


class Pooling(QuantumModule):
    """ 
    Quantum Pooling layer
    Reduces the dimension of the image by applying a succession of CNOT gates
    and discarding the control qubits.

    :param kernel_size: Dimension by which the image is reduced.
    """
    def __init__(self, dims: tuple[int], kernel_size: int | tuple[int] = 2):
        super().__init__()

        if isinstance(kernel_size, int):
            self._kernel_size = (kernel_size, kernel_size)
        else:
            self._kernel_size = kernel_size

        self._dims = dims

        if (dims[0] % self._kernel_size[0]) or (dims[1] % self._kernel_size[1]):
            raise ValueError("The dimension of the image should be divisible by the kernel size.")
        
        # The following permutation arranges the control qubits to the bottom of the register and
        # the target qubits to the top.
        _perm = []
        _keep, _bin = 0, dims[0] // self._kernel_size[0] + dims[1] // self._kernel_size[1]
        for d, k in zip(dims, self._kernel_size):
            for i in range(d):
                _perm.append(_keep if i % k == 0 else _bin)
                _keep += i % k == 0
                _bin += i % k != 0

        self._perm_u = np.zeros((2 ** sum(dims), 2 ** sum(dims)), dtype=np.int16)

        for i in range(2 ** sum(dims)):
            bit_string = format(i, f'0{sum(dims)}b')
            j = int(''.join(bit_string[_perm[j]] for j in range(sum(dims))), 2)
            self._perm_u[j, i] = 1


    def forward(self, qdev):
        # Pool X register
        for pool in reversed(range(self._dims[0] // self._kernel_size[0])):
            for i in range(self._kernel_size[0] - 1):
                qdev.cx([pool + i, pool + i + 1])

        # Pool Y register
        for pool in reversed(range(self._dims[1] // self._kernel_size[1])):
            for i in range(self._kernel_size[1] - 1):
                qdev.cx([pool + i, pool + i + 1])

        qdev.qubitunitary(
            wires=[*range(sum(self._dims))], 
            params=torch.tensor(self._perm_u)
        )


class PyramidDense(QuantumModule):
    """ 
    "Pyramidal" Dense Layer.
    RBS gates are applied between pairs of qubits in the merged register in a 
    pyramidal pattern.

    :param n_wires: Size of the dense layer
    """
    def __init__(self, n_wires: int):
        super().__init__() 

        half_pyramid = [
            [[2 * i, 2 * i + 1] for i in range(col - 1)] +
            [[2 * i + 1, 2 * i + 2] for i in range(col - 1)]
            for col in range(2, n_wires - 1)
        ]
        self._rbs_list = [RBS(wires=wires) for wires_list in half_pyramid for wires in wires_list]
        self._rbs_list += [RBS(wires=wires) for wires_list in reversed(half_pyramid[:-1]) for wires in wires_list]

    def forward(self, qdev):
        for rbs in self._rbs_list:
            rbs(qdev)
