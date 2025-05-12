import torch
import torch.nn as nn

import numpy as np
from scipy.special import comb
from itertools import combinations

from torchquantum import QuantumModule, QuantumDevice, partial_trace, matrix_form
from torchquantum.encoding import Encoder

class OneHotAmplitude(Encoder):
    """
    Encodes 2D data using one-hot amplitude encoding

    :param dims: Image dimensions.
    """
    def __init__(self, dims):
        super().__init__()

        self.dims = dims

        # Define basis states for each register.
        basisX = torch.eye(self.dims[0])
        basisY = torch.eye(self.dims[1])

        # Get cartesian product of indices
        idx_x, idx_y = torch.arange(basisX.size(0)), torch.arange(basisY.size(0))
        grid_x, grid_y = torch.meshgrid(idx_x, idx_y, indexing='ij')

        basis_states = torch.cat(
            (basisX[grid_x.flatten()], basisY[grid_y.flatten()]), dim=1
        ) 
        zero = torch.tensor([1, 0]) 
        one = torch.tensor([0, 1])

        # Vectorize this later. States must be given as tensor products.
        self._basis_states = torch.zeros(len(basis_states), 2 ** len(basis_states[0]))
        for i, vec in enumerate(basis_states):
            kron_ = torch.tensor([1])
            for val in vec:
                kron_ = torch.kron(kron_, one if val else zero)

            self._basis_states[i] = kron_

    def forward(self, qdev, X):
        if qdev.n_wires != sum(self.dims):
            raise ValueError(f'Quantum Device does not match image dimensions. {qdev.n_wires} != {sum(self.dims)}')
                
        if len(X.shape) != 3:
            X = X.unsqueeze(0)
        
        # Normalize image
        X = X.to(dtype=torch.complex64)
        norm = torch.sqrt(torch.square(X).sum(dim=(1, 2)))
        X = X / norm.view(-1, 1, 1)

        # Flatten each image
        X_flat = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

        # Encode image in amplitudes of each basis states
        X_state = torch.sum(self._basis_states.unsqueeze(0) * X_flat.unsqueeze(2), dim=1)

        qdev.set_states(X_state)

    def __repr__(self):
        return f'OneHotAmplitude({self.dims[0]}, {self.dims[1]})'


class RBS(QuantumModule):
    """ 
    Reconfigurable Beam splitter

    :param theta: RBS gate angle.
    :param wires: wires to apply the RBS between.
    """
    def __init__(self, theta = None, wires = None):
        super().__init__()

        # Instantiate theta as a torch tensor
        if theta is None:
            theta = 2 * np.pi * torch.rand(1)

        self.theta = torch.nn.Parameter(theta)
        self.wires = wires

    def forward(self, qdev: QuantumDevice, wires = None):
        if wires is None and self.wires is None:
            raise ValueError("Please specify wires to place RBS gate.")

        wires = wires if wires is not None else self.wires

        c = torch.cos(self.theta).item()
        s = torch.sin(self.theta).item()

        qdev.qubitunitary(
            wires=wires, 
            params=[[1, 0, 0, 0],
                    [0, c, s, 0],
                    [0, -s, c, 0],
                    [0, 0, 0, 1]]
        )

    def __repr__(self):
        if self.wires is not None:
            return f'RBS(wires=[{self.wires[0]}, {self.wires[1]}])'
        else:
            return 'RBS'

class Conv2dFilter(QuantumModule):
    """ 
    2D Convolutional filter applied to a register within a convolutional layer.
    Each convolutional filter consists of a series of RBS gates applied between 
    each pair of qubits in the register.

    :param kernel_size: Size of the convolutional filter
    """
    def __init__(self, kernel_size):
        super(Conv2dFilter, self).__init__()

        self.kernel_size = kernel_size
        num_rbs = (self.kernel_size // 2) * ((self.kernel_size + 1) // 2) + \
            ((self.kernel_size - 1) // 2) * ((self.kernel_size - 1) // 2)

        # Instantiate RBS gates
        self._rbs_list = torch.nn.ModuleList([RBS() for i in range(num_rbs)])

        # Determine wires to apply each RBS gate
        even_wires = [[2 * i, 2 * i + 1] for i in range(self.kernel_size // 2)] 
        odd_wires = [[2 * i + 1, 2 * i + 2] for i in range((self.kernel_size - 1) // 2)]

        rbs_wires = []
        for col in range((2 * self.kernel_size - 3)):
            wires = odd_wires if col % 2 else even_wires
            rbs_wires += wires

        self._rbs_wires = np.array(rbs_wires)


    def forward(self, qdev, start_wire = None):
        rbs_wires = self._rbs_wires + start_wire

        for i, rbs in enumerate(self._rbs_list):
            rbs(qdev, rbs_wires[i])

    def __repr__(self):
        return f'Conv2dFilter(kernel_size={self.kernel_size})'


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

        self.dims = dims
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        stride = kernel_size if stride is None else stride
        self.stride = (stride, stride) if isinstance(stride, int) else stride    

        self._filterX = Conv2dFilter(self.kernel_size[0])
        self._filterY = Conv2dFilter(self.kernel_size[1])

        self._start_wires_x = [
            i * self.stride[0] for i in range((dims[0] - self.kernel_size[0] + self.stride[0]) // self.stride[0])
            if i * self.stride[0] < dims[0]
        ]
        self._start_wires_y = [
            i * self.stride[1] + dims[0] for i in range((dims[1] - self.kernel_size[1] + self.stride[1]) // self.stride[1])
            if i * self.stride[1] < sum(dims) - self.kernel_size[1]
        ]
    
    def forward(self, qdev):
        for wire in self._start_wires_x:
            self._filterX(qdev, start_wire=wire)
        
        for wire in self._start_wires_y:
            self._filterY(qdev, start_wire=wire)

    def __repr__(self):
        return f'Conv2d(dims={self.dims}, kernel_size={self.kernel_size}, stride={self.stride})'


class Pooling(QuantumModule):
    """ 
    Quantum Pooling layer
    Reduces the dimension of the image by applying a succession of CNOT gates
    and discarding the control qubits.

    :param dims: Image dimensinos.
    :param kernel_size: Dimension by which the image is reduced.
    """
    def __init__(self, dims: tuple[int], kernel_size: int | tuple[int] = 2):
        super().__init__()

        self.dims = dims
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if (dims[0] % self.kernel_size[0]) or (dims[1] % self.kernel_size[1]):
            raise ValueError("The dimension of the image should be divisible by the kernel size.")
        
        # Determine wires in each register to perform CNOT gates
        registerX, registerY = list(range(dims[0])), list(range(dims[0], dims[0] + dims[1]))

        self._registerX_pooled = [
            registerX[i:i + kernel_size][::-1]
            for i in range(0, len(registerX), kernel_size)
        ]
        self._registerY_pooled = [
            registerY[i:i + kernel_size][::-1]
            for i in range(0, len(registerY), kernel_size)
        ]
        
        # The following permutation arranges the control qubits to the bottom of the register and target qubits to the top
        self._swaps = [[i, self.kernel_size[0] * i] for i in range(1, dims[0] // self.kernel_size[0])]
        self._swaps += [[dims[0] // self.kernel_size[0] + i, dims[0] + self.kernel_size[1] * i] for i in range(dims[1] // self.kernel_size[1])]

    def forward(self, qdev):
        # Pool X register
        for pool in reversed(self._registerX_pooled):
            for i in range(len(pool) - 1):
                qdev.cx([pool[i], pool[i + 1]])

        # Pool Y register
        for pool in reversed(self._registerY_pooled):
            for i in range(len(pool) - 1):
                qdev.cx([pool[i], pool[i + 1]])

        # Drop control qubits to bottom of register
        for swap in self._swaps:
            qdev.swap(swap)

    def __repr__(self):
        return f'Pooling(dims={self.dims}, kernel_size={self.kernel_size})'


class PyramidDense(QuantumModule):
    """ 
    "Pyramidal" Dense Layer.
    RBS gates are applied between pairs of qubits in the merged register in a 
    pyramidal pattern.

    :param n_wires: Size of the dense layer
    """
    def __init__(self, n_wires: int):
        super().__init__() 

        self.n_wires = n_wires

        wires_list = []
        for col in range(2 * n_wires - 3):
            if 2 * (col - 1) + 1 < n_wires:
                wires_list += [[2 * i, 2 * i + 1] for i in range(col)]
            if 2 * (col - 1) + 2 < n_wires:
                wires_list += [[2 * i + 1, 2 * i + 2] for i in range(col)]

        wires_list += list(reversed(wires_list[:-1]))
        self._rbs_list = torch.nn.ModuleList([RBS(wires=wires) for wires in wires_list])

    def forward(self, qdev):
        for rbs in self._rbs_list:
            rbs(qdev)

    def __repr__(self):
        return f'PyramidDense(n_wires={self.n_wires})'


class MeasureLayer(torch.nn.Module):
    """ 
    Measures the quantum device and outputs the results as a layer.

    :param n_wires: Subset of qubits to be measured.
    """
    def __init__(self, n_wires):
        super().__init__()

        self._n_wires = n_wires
        self._mask = (list(self._generate_mask()))

    def forward(self, qdev):
        density_matrix = matrix_form(partial_trace(qdev, keep_indices=list(range(self._n_wires))))
        
        results = density_matrix.diagonal(dim1=1, dim2=2)
        filtered_results = torch.abs(results[:, self._mask])

        return filtered_results

    def output_states(self):
        """Generate all possible bit strings with two 1s."""
        if self._n_wires < 2:
            return
        for i, j in combinations(range(self._n_wires), 2):
            bit_string = ['0'] * self._n_wires
            bit_string[i] = '1'
            bit_string[j] = '1'
            yield ''.join(bit_string)

    def _generate_mask(self):
        """Generates mask that filters states with hamming weights
        not equal to 2."""
        for i in range(2 ** self._n_wires):
            bit_string = format(i, f'0{self._n_wires}b')
            if bit_string.count('1') == 2:
                yield True
            else:
                yield False

    def __repr__(self):
        return f"Measure(n_wires={self._n_wires})"