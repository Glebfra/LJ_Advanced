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
    @cuda.jit
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


class GpuVector(GpuOperations):
    def __init__(self, vector: dict):
        self.vector = vector
        self.basis = [axis for axis in list(vector.keys()) if axis != 'size']
        self.size = vector['size']
        self.dimension = len(self.basis)

        self.device = cuda.get_current_device()
        self.tpb = self.device.WARP_SIZE
        self.bpg = int(np.ceil(self.size / self.tpb))

    @classmethod
    def create_vector_from_dict(cls, vector: dict):
        """
        This method creates the GpuVector from dictionary
        :param vector: The dictionary with numpy arrays
        :return: GpuVector
        """
        basis = list(vector.keys())
        vector['size'] = vector[basis[0]].size
        for axis in basis:
            vector[axis] = cuda.to_device(vector[axis])
        return cls(vector)

    def __add__(self, other):
        answer = {axis: cuda.device_array_like(other.vector[self.basis[0]]) for axis in self.basis}
        answer['size'] = self.size
        for axis in self.basis:
            self.cuda_addition[self.bpg, self.tpb](self.vector[axis], other.vector[axis], answer[axis])
        return GpuVector(answer)

    __radd__ = __add__

    def __sub__(self, other):
        answer = {axis: cuda.device_array_like(other.vector[self.basis[0]]) for axis in self.basis}
        answer['size'] = self.size
        for axis in self.basis:
            self.cuda_substraction[self.bpg, self.tpb](self.vector[axis], other.vector[axis], answer[axis])
        return GpuVector(answer)

    def __rsub__(self, other):
        answer = {axis: cuda.device_array_like(other.vector[self.basis[0]]) for axis in self.basis}
        answer['size'] = self.size
        for axis in self.basis:
            self.cuda_substraction[self.bpg, self.tpb](other.vector[axis], self.vector[axis], answer[axis])
        return GpuVector(answer)

    def __mul__(self, other):
        answer = {axis: cuda.device_array_like(other.vector[self.basis[0]]) for axis in self.basis}
        answer['size'] = self.size
        for axis in self.basis:
            self.cuda_multiplication[self.bpg, self.tpb](self.vector[axis], other.vector[axis], answer[axis])
        return GpuVector(answer)

    __rmul__ = __mul__

    def __truediv__(self, other):
        answer = {axis: cuda.device_array_like(other.vector[self.basis[0]]) for axis in self.basis}
        answer['size'] = self.size
        for axis in self.basis:
            self.cuda_divide[self.bpg, self.tpb](self.vector[axis], other.vector[axis], answer[axis])
        return GpuVector(answer)

    def __rtruediv__(self, other):
        answer = {axis: cuda.device_array_like(other.vector[self.basis[0]]) for axis in self.basis}
        answer['size'] = self.size
        for axis in self.basis:
            self.cuda_divide[self.bpg, self.tpb](other.vector[axis], self.vector[axis], answer[axis])
        return GpuVector(answer)

    def __pow__(self, power, modulo=None):
        answer = {axis: cuda.device_array_like(self.vector[self.basis[0]]) for axis in self.basis}
        answer['size'] = self.size
        for axis in self.basis:
            self.cuda_power[self.bpg, self.tpb](self.vector[axis], power, answer[axis])
        return GpuVector(answer)

    def __str__(self):
        return str(self.to_dict())

    def __len__(self) -> int:
        return self.size

    def __abs__(self):
        pass

    def to_dict(self):
        for axis in self.basis:
            self.vector[axis] = self.vector[axis].copy_to_host()
        return self.vector


if __name__ == '__main__':
    a = GpuVector.create_vector_from_dict({axis: np.random.sample((1000, 1000)) for axis in 'xyz'})
    b = GpuVector.create_vector_from_dict({axis: np.random.sample((1000, 1000)) for axis in 'xyz'})
    c = 5
    d = (a + b) ** c
    print(d)
