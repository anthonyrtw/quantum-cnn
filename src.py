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
        self._basisX = torch.eye(2 ** self._dims[0]).unsqueeze(0)  # Shape: (1, 2**dim_x, 2**dim_x)
        self._basisY = torch.eye(2 ** self._dims[1]).unsqueeze(0)  # Shape: (1, 2**dim_y, 2**dim_y)

    def forward(self, qdev, X):
        if qdev.n_wires != sum(self._dims):
            raise ValueError('Quantum Device does not match image dimensions.')

        bsz = X.shape[0]
        device = X.device

        state = torch.zeros((bsz, 2 ** qdev.n_wires), device=device, dtype=X.dtype)
        norm = 1 / torch.sqrt(torch.sum(X, dim=(1, 2), keepdim=True))

        for bsz in range(X.shape[0]):
            for x in range(X.shape[1]):
                for y in range(X.shape[2]):
                    kron_product = torch.kron(self._basisX[0][x], self._basisY[0][y]).to(device)
                    state += (norm.squeeze(-1).squeeze(-1) * X[:, x, y])[:, None] * kron_product[None, :]

        qdev.set_states(state) 


class RBS(QuantumModule):
    """ 
    Reconfigurable Beam splitter

    :param theta: Angle in
    :param wires: wires to apply RBS
    """
    def __init__(self, theta = None, wires = None):
        super().__init__()

        if isinstance(theta, (float, int)):
            self._theta = nn.Parameter(torch.tensor([theta]))
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

        wires1 = [
            [2 * i, 2 * i + 1]
            for i in range(kernel_size // 2)
        ]
        wires2 = [
            [2 * i + 1, 2 * i + 2]
            for i in range((kernel_size - 1) // 2)
        ]

        for _ in range((kernel_size + 1) // 2):
            self._rbs_list += [
                RBS(wires=wires) for wires in wires1
            ]
            self._rbs_list += [
                RBS(wires=wires) for wires in wires2
            ]
    def forward(self, qdev):
        for rbs in self._rbs_list:
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
    def __init__(self, dims: tuple, kernel_size: int | tuple[int], stride: int = None):
        super().__init__()

        if isinstance(kernel_size, int):
            self._kernel_size = [kernel_size, kernel_size]
        else:
            self._kernel_size = kernel_size

        self._dims = dims
        self._kernel_size= kernel_size
        self._stride = kernel_size if stride is None else stride

        self._filterX = Conv2dFilter(kernel_size)
        self._filterY = Conv2dFilter(kernel_size)

        self._wires_x = [
            [*range(i * self._stride, self._kernel_size[0] + i * self._stride)]
            for i in range(self._dims[0] - self._kernel_size[0] + 1)
        ]
        self._wires_y = [
            [*range(self._dims[0] + i * self._stride, self._dims[0] + self._kernel_size[0] + i * self._stride)]
            for i in range(self._dims[1] - self._kernel_size[0] + 1)
        ]

    def forward(self, qdev):
        for wires in self._wires_x:
            RBS(qdev, wires=wires)
        
        for wires in self._wires_y:
            RBS(qdev, wires=wires)


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
            self._kernel_size = [kernel_size, kernel_size]
        else:
            self._kernel_size = kernel_size

        self._dims = dims

        if not (dims[0] % kernel_size) or not (dims[1] % kernel_size[1]):
            raise ValueError("The dimension of the image should be divisible by the kernel size.")
        
        # The following permutation arranges the control qubits to the bottom of the register and
        # the target qubits to the top.
        _perm = []
        _keep, _bin = 0, dims[0] // kernel_size[0] + dims[1] // kernel_size[1]
        for d, k in zip(dims, kernel_size):
            for i in range(d):
                _perm.append(_keep if i % k == 0 else _bin)
                _keep += i % k == 0
                _bin += i % k != 0
        self._perm_u = np.zeros((sum(dims), sum(dims)))
        for i, v in enumerate(_perm):
            self._perm_u[v, i] = 1

    def forward(self, qdev):
        # Pool X register
        for pool in reversed(range(self._dims[0] // self._kernel_size[0])):
            for i in range(self._kernel_size[0] - 1):
                qdev.cx(pool + i, i + 1)

        # Pool Y register
        for pool in reversed(range(self._dims[1] // self._kernel_size[1])):
            for i in range(self._kernel_size[1] - 1):
                qdev.cx(pool + i, i + 1)

        qdev.qubitunitary(
            wires=[*range(sum(self._dims))], 
            params=self._perm_u
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
