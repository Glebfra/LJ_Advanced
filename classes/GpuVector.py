import numpy as np
from numba import cuda


def types(func):
    """
    This decorator looking for types and uses the special math operations to create the GpuVector from them
    :param func: Decorating function
    :return: Decorated function
    """

    def inner(*args, **kwargs):
        if type(args[1]) != GpuVector:
            rows = args[0].vector[args[0].basis[0]].shape[0]
            columns = args[0].vector[args[0].basis[0]].shape[1]
            other = GpuVector.create_vector_from_dict(
                {axis: np.ones((rows, columns)) * args[1] for axis in args[0].basis})
            return func(args[0], other, **kwargs)
        return func(*args, **kwargs)

    return inner


class GpuOperations(object):
    @staticmethod
    @cuda.jit(fastmath=True)
    def cuda_addition(A, B, C):
        row, column = cuda.grid(2)
        if row < C.shape[0] and column < C.shape[1]:
            C[row, column] = A[row, column] + B[row, column]

    @staticmethod
    @cuda.jit(fastmath=True)
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
    @cuda.jit(fastmath=True)
    def cuda_divide(A, B, C):
        row, column = cuda.grid(2)
        if row < C.shape[0] and column < C.shape[1]:
            C[row, column] = A[row, column] / B[row, column]

    @staticmethod
    @cuda.jit(fastmath=True)
    def cuda_floor_divide(A, B, C):
        row, column = cuda.grid(2)
        if row < C.shape[0] and column < C.shape[1]:
            C[row, column] = A[row, column] // B[row, column]

    @staticmethod
    @cuda.jit(fastmath=True)
    def cuda_power(A, B, C):
        row, column = cuda.grid(2)
        if row < C.shape[0] and column < C.shape[1]:
            C[row, column] = A[row, column] ** B

    @staticmethod
    @cuda.jit(fastmath=True)
    def cuda_matrix_mul(A, B, C):
        row, column = cuda.grid(2)
        if row < A.shape[0] and column < B.shape[1]:
            temp = 0
            for k in range(A.shape[1]):
                temp += A[row, k] * B[k, column]
            C[row, column] = temp

    @staticmethod
    @cuda.jit(fastmath=True)
    def cuda_sum(A, B):
        row, column = cuda.grid(2)
        temp = 0
        if row < A.shape[0] and column < A.shape[1]:
            temp += A[row, column]
        B = temp

    @staticmethod
    @cuda.jit(fastmath=True)
    def cuda_sum_columns(A, B):
        row, column = cuda.grid(2)
        temp = 0
        if row < A.shape[0]:
            for j in range(A.shape[1]):
                temp += A[row, j]
            B[row, 0] = temp


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

    @types
    def __add__(self, other):
        for axis in self.basis:
            self.cuda_addition[self.bpg, self.tpb](self.vector[axis], other.vector[axis], self.answer[axis])
        return GpuVector(self.answer)

    __radd__ = __add__

    __iadd__ = __add__

    @types
    def __sub__(self, other):
        for axis in self.basis:
            self.cuda_substraction[self.bpg, self.tpb](self.vector[axis], other.vector[axis], self.answer[axis])
        return GpuVector(self.answer)

    @types
    def __rsub__(self, other):
        for axis in self.basis:
            self.cuda_substraction[self.bpg, self.tpb](other.vector[axis], self.vector[axis], self.answer[axis])
        return GpuVector(self.answer)

    __isub__ = __sub__

    @types
    def __mul__(self, other):
        for axis in self.basis:
            self.cuda_multiplication[self.bpg, self.tpb](self.vector[axis], other.vector[axis], self.answer[axis])
        return GpuVector(self.answer)

    __rmul__ = __mul__

    __imul__ = __mul__

    @types
    def __matmul__(self, other):
        answer = {axis: cuda.to_device(np.zeros((self.vector[axis].shape[0], other.vector[axis].shape[1]))) for axis in
                  self.basis}
        for axis in self.basis:
            self.cuda_matrix_mul[self.bpg, self.tpb](self.vector[axis], other.vector[axis], answer[axis])
        return GpuVector(answer)

    @types
    def __rmatmul__(self, other):
        answer = {axis: cuda.to_device(np.zeros((self.size, self.size))) for axis in self.basis}
        for axis in self.basis:
            self.cuda_matrix_mul[self.bpg, self.tpb](other.vector[axis], self.vector[axis], answer[axis])
        return GpuVector(answer)

    @types
    def __truediv__(self, other):
        for axis in self.basis:
            self.cuda_divide[self.bpg, self.tpb](self.vector[axis], other.vector[axis], self.answer[axis])
        return GpuVector(self.answer)

    @types
    def __rtruediv__(self, other):
        for axis in self.basis:
            self.cuda_divide[self.bpg, self.tpb](other.vector[axis], self.vector[axis], self.answer[axis])
        return GpuVector(self.answer)

    @types
    def __floordiv__(self, other):
        for axis in self.basis:
            self.cuda_floor_divide[self.bpg, self.tpb](self.vector[axis], other.vector[axis], self.answer[axis])
        return GpuVector(self.answer)

    @types
    def __rfloordiv__(self, other):
        for axis in self.basis:
            self.cuda_floor_divide[self.bpg, self.tpb](other.vector[axis], self.vector[axis], self.answer[axis])
        return GpuVector(self.answer)

    def __pow__(self, power):
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
        """This method is converting the GpuVector to dictionary"""
        for axis in self.basis:
            self.vector[axis] = self.vector[axis].copy_to_host()
        return self.vector

    @property
    def T(self):
        """This property method trasponse the GpuVector"""
        self.vector = self.to_dict()
        self.vector = {axis: cuda.to_device(self.vector[axis].T) for axis in self.basis}
        return GpuVector(self.vector)

    def differences(self):
        ones_matrix = cuda.to_device(np.ones((1, self.size)))
        e = cuda.to_device(np.eye(self.size))
        mat_mul_1, mat_mul_2 = cuda.device_array_like(e), cuda.device_array_like(e)
        differences = {axis: cuda.device_array_like(e) for axis in self.basis}

        for axis in self.basis:
            self.cuda_matrix_mul[self.bpg, self.tpb](self.vector[axis], ones_matrix, mat_mul_1)
            self.cuda_matrix_mul[self.bpg, self.tpb](ones_matrix.T, self.vector[axis].T, mat_mul_2)
            self.cuda_substraction[self.bpg, self.tpb](mat_mul_1, mat_mul_2, differences[axis])
            self.cuda_addition[self.bpg, self.tpb](differences[axis], e, differences[axis])
        return GpuVector(differences)

    def get_keys(self) -> list:
        return self.basis

    def sum(self) -> float:
        temp = 0
        for axis in self.basis:
            temp += self.vector[axis].copy_to_host().sum()
        return temp

    def sum_columns(self):
        answer = {axis: cuda.to_device(np.zeros((self.vector[self.basis[0]].shape[0], 1))) for axis in self.basis}
        for axis in self.basis:
            self.cuda_sum_columns[self.bpg, self.tpb](self.vector[axis], answer[axis])
        return GpuVector(answer)


if __name__ == '__main__':
    a = GpuVector.create_vector_from_dict({axis: np.ones((1024, 1)) for axis in 'xyz'})
    b = 2
    print(a + b)
