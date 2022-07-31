import numpy as np
from numba import cuda

from classes.Gpu import GpuOperations


class Vector(GpuOperations):
    def __init__(self, vector: dict):
        self.vector = vector
        self.basis = [axis for axis in list(vector.keys()) if axis != 'size']
        self.size = vector['size']

        self.device = cuda.get_current_device()
        self.tpb = self.device.WARP_SIZE
        self.bpg = int(np.ceil(self.size / self.tpb))

    @classmethod
    def create_vector_from_dict(cls, vector: dict):
        """
        This method creates the GpuVector from dictionary
        :param vector: The dictionary with numpy arrays
        :return: Vector
        """
        basis = list(vector.keys())
        vector['size'] = vector[basis[0]].size
        for axis in basis:
            vector[axis] = cuda.to_device(vector[axis])
        return cls(vector)

    def __add__(self, other):
        answer = cuda.device_array_like(other.vector[self.basis[0]])
        for axis in self.basis:
            self.cuda_addition[self.bpg, self.tpb](self.vector[axis], other.vector[axis], answer[axis])
        return Vector(answer)

    def to_dict(self):
        for axis in self.basis:
            self.vector[axis] = self.vector[axis].copy_to_host()
        return self.vector


if __name__ == '__main__':
    a = Vector.create_vector_from_dict({axis: np.random.sample((1000, 1)) for axis in 'xyz'})
    b = Vector.create_vector_from_dict({axis: np.random.sample((1000, 1)) for axis in 'xyz'})
    c = a + b
    print(c.to_dict())
