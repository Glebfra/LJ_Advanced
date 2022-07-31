import numpy as np

from classes.Gpu import GpuOperations


class Vector(GpuOperations):
    def __init__(self, vector: dict) -> None:
        self.basis = list(vector.keys())
        self.vector = vector
        temp = vector[self.basis[0]]
        self.length = len(temp) if type(temp) == np.ndarray else 1

    def __add__(self, other):
        return self._arithmetic_operation(other, operation='+')

    __radd__ = __add__

    def __sub__(self, other):
        return self._arithmetic_operation(other, operation='-')

    def __rsub__(self, other):
        return self._arithmetic_operation(other, operation='-', inverse=True)

    def __mul__(self, other):
        return self._arithmetic_operation(other, operation='*')

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._arithmetic_operation(other, operation='/')

    def __rtruediv__(self, other):
        return self._arithmetic_operation(other, operation='/', inverse=True)

    def __floordiv__(self, other):
        return self._arithmetic_operation(other, operation='//')

    def __rfloordiv__(self, other):
        return self._arithmetic_operation(other, operation='//', inverse=True)

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

    def differences(self):
        differences = {}
        ones_matrix = np.ones((1, self.length))
        e = np.eye(self.length)
        for axis in self.basis:
            differences[axis] = self.vector[axis] @ ones_matrix - ones_matrix.T @ self.vector[axis].T
            differences[axis] += e
        return Vector(differences)

    def _arithmetic_operation(self, other, operation, inverse=False):
        temp = {}
        if type(other) == Vector:
            if not inverse:
                for axis in self.basis:
                    temp[axis] = eval(f'self.vector[axis] {operation} other.vector[axis]')
            else:
                for axis in self.basis:
                    temp[axis] = eval(f'other.vector[axis] {operation} self.vector[axis]')
        elif type(other) == np.ndarray or int or float:
            if not inverse:
                for axis in self.basis:
                    temp[axis] = eval(f'self.vector[axis] {operation} other')
            else:
                for axis in self.basis:
                    temp[axis] = eval(f'other {operation} self.vector[axis]')
        else:
            raise TypeError('Incompatible type use Vector, ndarray, int or float')
        return Vector(temp)


if __name__ == '__main__':
    a = Vector({axis: np.random.sample((1000, 1)) for axis in 'xyz'})
    b = Vector({axis: np.random.sample((1000, 1)) for axis in 'xyz'})
    print((a + b).to_dict())
