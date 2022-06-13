import numpy as np


class LJ(object):
    def __init__(self, sigma: float, eps: float) -> None:
        self.sigma = sigma
        self.eps = eps

    @staticmethod
    def calculate_absolute_radius(radius_differences: dict) -> np.ndarray:
        r = 0
        for axis in radius_differences:
            r += radius_differences[axis] ** 2
        r = np.sqrt(r)

        E = np.eye(len(radius_differences['x']))
        r += E
        return r

    @classmethod
    def create_default_LJ(cls):
        return cls(sigma=1e-10, eps=120 * 1.38e-23)

    def potential(self, radius_differences: dict) -> float:
        r = self.calculate_absolute_radius(radius_differences)
        temp = self.sigma / r
        return (4 * self.eps * (temp ** 12 - temp ** 6)).sum() / 2

    def force(self, radius_differences: dict) -> dict:
        force = {}
        r = self.calculate_absolute_radius(radius_differences)
        temp = self.sigma / r
        for axis in radius_differences:
            force[axis] = (
                    8 * self.eps * (12 * temp ** 14 - 6 * temp ** 8) * radius_differences[axis] / self.sigma ** 2) \
                .sum(1).reshape((len(radius_differences['x']), 1))
        return force
