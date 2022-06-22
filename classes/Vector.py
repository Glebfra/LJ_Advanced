import numpy as np


class Vector(object):
    def __init__(self, vector: dict):
        self.basis = list(vector.keys())
        self.vector = vector
        self.length = len(self.vector[self.basis[0]])

    def __add__(self, other):
        temp = {}
        if type(other) == dict:
            for axis in self.basis:
                temp[axis] = self.vector[axis] + other[axis]
                continue
        elif type(other) == Vector:
            for axis in self.basis:
                temp[axis] = self.vector[axis] + other.vector[axis]
                continue
        elif type(other) == int or float:
            for axis in self.basis:
                temp[axis] = self.vector[axis] + other
                continue
        return Vector(temp)

    def __sub__(self, other):
        temp = {}
        if type(other) == dict:
            for axis in self.basis:
                temp[axis] = self.vector[axis] - other[axis]
                continue
        elif type(other) == Vector:
            for axis in self.basis:
                temp[axis] = self.vector[axis] - other.vector[axis]
                continue
        elif type(other) == int or float:
            for axis in self.basis:
                temp[axis] = self.vector[axis] - other
                continue
        return Vector(temp)

    def __mul__(self, other):
        temp = {}
        if type(other) == dict:
            for axis in self.basis:
                temp[axis] = self.vector[axis] * other[axis]
                continue
        elif type(other) == Vector:
            for axis in self.basis:
                temp[axis] = self.vector[axis] * other.vector[axis]
                continue
        elif type(other) == int or float:
            for axis in self.basis:
                temp[axis] = self.vector[axis] * other
                continue
        return Vector(temp)

    def __truediv__(self, other):
        temp = {}
        if type(other) == dict:
            for axis in self.basis:
                temp[axis] = self.vector[axis] / other[axis]
                continue
        elif type(other) == Vector:
            for axis in self.basis:
                temp[axis] = self.vector[axis] / other.vector[axis]
                continue
        elif type(other) == int or float:
            for axis in self.basis:
                temp[axis] = self.vector[axis] / other
                continue
        return Vector(temp)

    def __pow__(self, power, modulo=None):
        temp = {}
        for axis in self.basis:
            temp[axis] = self.vector[axis] ** power
        return Vector(temp)

    def __len__(self) -> int:
        return self.length

    def __abs__(self):
        temp = 0
        for axis in self.basis:
            temp += self.vector[axis] ** 2
        temp = np.sqrt(temp)
        temp += np.eye(self.length)
        return temp

    def get_keys(self):
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

    def to_dict(self):
        return self.vector
