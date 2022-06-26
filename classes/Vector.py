import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import pycuda.autoinit


class Vector(object):
    def __init__(self, vector: dict) -> None:
        self.basis = list(vector.keys())
        self.vector = vector
        temp = self.vector[self.basis[0]]
        self.length = len(temp) if type(temp) == np.ndarray else 1

    def __add__(self, other):
        temp = {}
        if type(other) == Vector:
            for axis in self.basis:
                temp[axis] = self.vector[axis] + other.vector[axis]
        elif type(other) == np.ndarray:
            for axis in self.basis:
                temp[axis] = self.vector[axis] + other
        elif type(other) == int or float:
            for axis in self.basis:
                temp[axis] = self.vector[axis] + other
        return Vector(temp)

    __radd__ = __add__

    def __sub__(self, other):
        temp = {}
        if type(other) == dict:
            for axis in self.basis:
                temp[axis] = self.vector[axis] - other[axis]
        elif type(other) == Vector:
            for axis in self.basis:
                temp[axis] = self.vector[axis] - other.vector[axis]
        elif type(other) == np.ndarray:
            for axis in self.basis:
                temp[axis] = self.vector[axis] - other
        elif type(other) == int or float:
            for axis in self.basis:
                temp[axis] = self.vector[axis] - other
        return Vector(temp)

    def __rsub__(self, other):
        temp = {}
        if type(other) == dict:
            for axis in self.basis:
                temp[axis] = other[axis] - self.vector[axis]
        elif type(other) == Vector:
            for axis in self.basis:
                temp[axis] = other[axis] - self.vector[axis]
        elif type(other) == np.ndarray:
            for axis in self.basis:
                temp[axis] = other - self.vector[axis]
        elif type(other) == int or float:
            for axis in self.basis:
                temp[axis] = other - self.vector[axis]
        return Vector(temp)

    def __mul__(self, other):
        temp = {}
        if type(other) == dict:
            for axis in self.basis:
                temp[axis] = self.vector[axis] * other[axis]
        elif type(other) == Vector:
            for axis in self.basis:
                temp[axis] = self.vector[axis] * other.vector[axis]
        elif type(other) == np.ndarray:
            for axis in self.basis:
                temp[axis] = self.vector[axis] * other
        elif type(other) == int or float:
            for axis in self.basis:
                temp[axis] = self.vector[axis] * other
        return Vector(temp)

    __rmul__ = __mul__

    def __truediv__(self, other):
        temp = {}
        if type(other) == dict:
            for axis in self.basis:
                temp[axis] = self.vector[axis] / other[axis]
        elif type(other) == Vector:
            for axis in self.basis:
                temp[axis] = self.vector[axis] / other.vector[axis]
        elif type(other) == np.ndarray:
            for axis in self.basis:
                temp[axis] = self.vector[axis] / other
        elif type(other) == int or float:
            for axis in self.basis:
                temp[axis] = self.vector[axis] / other
        return Vector(temp)

    def __rtruediv__(self, other):
        temp = {}
        if type(other) == dict:
            for axis in self.basis:
                temp[axis] = other[axis] / self.vector[axis]
        elif type(other) == Vector:
            for axis in self.basis:
                temp[axis] = other[axis] / self.vector[axis]
        elif type(other) == np.ndarray:
            for axis in self.basis:
                temp[axis] = other / self.vector[axis]
        elif type(other) == int or float:
            for axis in self.basis:
                temp[axis] = other / self.vector[axis]
        return Vector(temp)

    def __floordiv__(self, other):
        temp = {}
        if type(other) == dict:
            for axis in self.basis:
                temp[axis] = self.vector[axis] // other[axis]
        elif type(other) == Vector:
            for axis in self.basis:
                temp[axis] = self.vector[axis] // other.vector[axis]
        elif type(other) == np.ndarray:
            for axis in self.basis:
                temp[axis] = self.vector[axis] // other
        elif type(other) == int or float:
            for axis in self.basis:
                temp[axis] = self.vector[axis] // other
        return Vector(temp)

    def __rfloordiv__(self, other):
        temp = {}
        if type(other) == dict:
            for axis in self.basis:
                temp[axis] = other[axis] // self.vector[axis]
        elif type(other) == Vector:
            for axis in self.basis:
                temp[axis] = other[axis] // self.vector[axis]
        elif type(other) == np.ndarray:
            for axis in self.basis:
                temp[axis] = other // self.vector[axis]
        elif type(other) == int or float:
            for axis in self.basis:
                temp[axis] = other // self.vector[axis]
        return Vector(temp)

    def __pow__(self, power, modulo=None):
        temp = {}
        if type(power) == int or float:
            for axis in self.basis:
                temp[axis] = self.vector[axis] ** power
        else:
            raise Exception(f"You cannot raise to this power ({power})")
        return Vector(temp)

    def __len__(self) -> int:
        return self.length

    def __abs__(self) -> np.ndarray:
        temp = 0
        for axis in self.basis:
            temp += self.vector[axis] ** 2
        temp = np.sqrt(temp)
        return temp

    def get_keys(self) -> list:
        return self.basis

    def sum_columns(self):
        temp = {}
        for axis in self.basis:
            temp[axis] = self.vector[axis].sum(1).reshape((self.length, 1))
        return Vector(temp)

    def sum_rows(self):
        temp = {}
        for axis in self.basis:
            temp[axis] = self.vector[axis].sum(0).reshape((1, self.length))
        return Vector(temp)

    def sum(self) -> float:
        temp = 0
        for axis in self.basis:
            temp += self.vector[axis].sum()
        return temp

    def to_dict(self) -> dict:
        return self.vector

    def get_average(self) -> float:
        return Vector(self.vector).sum() / (self.length * 3)


def _other_to_gpuarray(other: Vector):
    temp = {}
    if type(other) == Vector:
        for axis in other.get_keys():
            temp[axis] = gpuarray.to_gpu(other.vector[axis])
    elif type(other) == int or float:
        temp = gpuarray.to_gpu(other)
    return temp


if __name__ == '__main__':
    size = 1e7
    a = np.linspace (1, size) . astype (np.float32 )
    X_GPU = gpuarray.to_gpu(a)
    Y_GPU = cumath.sin(X_GPU)
    Y = Y_GPU.get()
    print(Y)
