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


if __name__ == '__main__':
    size = int(1e3)
    a = np.random.sample((size, size))
    b = np.random.sample((size, size))

    device = cuda.get_current_device()

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array_like(a)

    tpb = device.WARP_SIZE
    bpg = int(np.ceil(size / tpb))

    GpuOperations.cuda_divide[bpg, tpb](d_a, d_b, d_c)
    c = d_c.copy_to_host()
    print(c)
