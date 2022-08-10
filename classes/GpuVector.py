import numpy as np
from numba import cuda


class GpuOperations(object):
    @staticmethod
    @cuda.jit
    def cuda_addition(A, B, C):
        row, column = cuda.grid(2)
        if row < C.shape[0] and column < C.shape[1]:
            C[row, column] = A[row, column] + B[row, column]

    @staticmethod
    @cuda.jit
    def cuda_substraction(A, B, C):
        row, column = cuda.grid(2)
        if row < C.shape[0] and column < C.shape[1]:
            C[row, column] = A[row, column] - B[row, column]

    @staticmethod
    @cuda.jit(fastmath=True)
    def cuda_multiplication(A, B, C):
        row, column = cuda.grid(2)
        if row < C.shape[0] and column < C.shape[1]:
            C[row, column] = A[row, column] * B[row, column]

    @staticmethod
    @cuda.jit
    def cuda_divide(A, B, C):
        row, column = cuda.grid(2)
        if row < C.shape[0] and column < C.shape[1]:
            C[row, column] = A[row, column] / B[row, column]

    @staticmethod
    @cuda.jit
    def cuda_power(A, B, C):
        row, column = cuda.grid(2)
        if row < C.shape[0] and column < C.shape[1]:
            C[row, column] = A[row, column] ** B

    @staticmethod
    @cuda.jit
    def cuda_matrix_mul(A, B, C):
        row, column = cuda.grid(2)
        if row < A.shape[0] and column < B.shape[1]:
            temp = 0
            for k in range(A.shape[1]):
                temp += A[row, k] * B[k, column]
            C[row, column] = temp


class GpuVector(GpuOperations):
    def __init__(self, vector: dict):
        self.vector = vector
        self.basis = [axis for axis in list(vector.keys())]
        self.dimension = len(self.basis)
        self.size = self.vector[self.basis[0]].size

        # Creating the answer GpuArray
        self.answer = {axis: cuda.device_array_like(self.vector[self.basis[0]]) for axis in self.basis}

        # Getting the information from VideoCard and getting the threads per block (tpb) and blocks per grid(bpg)
        self.device = cuda.get_current_device()
        self.tpb = (self.device.WARP_SIZE, self.device.WARP_SIZE)
        bpg_1 = int(np.ceil(self.vector[self.basis[0]].shape[0] / self.tpb[0]))
        bpg_2 = int(np.ceil(self.vector[self.basis[0]].shape[1] / self.tpb[1]))
        self.bpg = (bpg_1, bpg_2)

    @classmethod
    def create_vector_from_dict(cls, vector: dict):
        """
        This method creates the GpuVector from dictionary
        :param vector: The dictionary with numpy arrays
        :return: GpuVector
        """
        basis = list(vector.keys())
        for axis in basis:
            vector[axis] = cuda.to_device(vector[axis])
        return cls(vector)

    def __add__(self, other):
        for axis in self.basis:
            self.cuda_addition[self.bpg, self.tpb](self.vector[axis], other.vector[axis], self.answer[axis])
        return GpuVector(self.answer)

    __radd__ = __add__

    def __sub__(self, other):
        for axis in self.basis:
            self.cuda_substraction[self.bpg, self.tpb](self.vector[axis], other.vector[axis], self.answer[axis])
        return GpuVector(self.answer)

    def __rsub__(self, other):
        for axis in self.basis:
            self.cuda_substraction[self.bpg, self.tpb](other.vector[axis], self.vector[axis], self.answer[axis])
        return GpuVector(self.answer)

    def __mul__(self, other):
        for axis in self.basis:
            self.cuda_multiplication[self.bpg, self.tpb](self.vector[axis], other.vector[axis], self.answer[axis])
        return GpuVector(self.answer)

    __rmul__ = __mul__

    def __matmul__(self, other):
        answer = {axis: cuda.to_device(np.zeros((self.vector[axis].shape[0], other.vector[axis].shape[1]))) for axis in
                  self.basis}
        for axis in self.basis:
            self.cuda_matrix_mul[self.bpg, self.tpb](self.vector[axis], other.vector[axis], answer[axis])
        return GpuVector(answer)

    def __rmatmul__(self, other):
        answer = {axis: cuda.to_device(np.zeros((self.size, self.size))) for axis in self.basis}
        for axis in self.basis:
            self.cuda_matrix_mul[self.bpg, self.tpb](other.vector[axis], self.vector[axis], answer[axis])
        return GpuVector(answer)

    def __truediv__(self, other):
        for axis in self.basis:
            self.cuda_divide[self.bpg, self.tpb](self.vector[axis], other.vector[axis], self.answer[axis])
        return GpuVector(self.answer)

    def __rtruediv__(self, other):
        for axis in self.basis:
            self.cuda_divide[self.bpg, self.tpb](other.vector[axis], self.vector[axis], self.answer[axis])
        return GpuVector(self.answer)

    def __pow__(self, power, modulo=None):
        for axis in self.basis:
            self.cuda_power[self.bpg, self.tpb](self.vector[axis], power, self.answer[axis])
        return GpuVector(self.answer)

    def __str__(self):
        return str(self.to_dict())

    def __len__(self) -> int:
        return self.size

    def __abs__(self):
        temp = cuda.device_array_like(self.vector[self.basis[0]])
        for axis in self.basis:
            self.cuda_power[self.bpg, self.tpb](self.vector[axis], 2, self.answer[axis])
            self.cuda_addition[self.bpg, self.tpb](temp, self.vector[axis], temp)
        return temp.copy_to_host()

    def to_dict(self):
        for axis in self.basis:
            self.vector[axis] = self.vector[axis].copy_to_host()
        return self.vector

    @property
    def T(self):
        self.vector = self.to_dict()
        self.vector = {axis: cuda.to_device(self.vector[axis].T) for axis in self.basis}
        return GpuVector(self.vector)

    def differences(self):
        ones_matrix = GpuVector.create_vector_from_dict({axis: np.ones((1, self.size)) for axis in self.basis})
        vector = GpuVector(self.vector)
        e = GpuVector.create_vector_from_dict({axis: np.eye(self.size) for axis in self.basis})

        differences = vector @ ones_matrix - ones_matrix.T @ vector.T
        differences = differences + e
        return differences


if __name__ == '__main__':
    a = GpuVector.create_vector_from_dict({axis: np.random.sample((32, 32)) for axis in 'xyz'})
    b = GpuVector.create_vector_from_dict({axis: np.random.sample((32, 32)) for axis in 'xyz'})
    print(a @ b)
