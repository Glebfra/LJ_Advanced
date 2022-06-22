import numpy as np
import numba
from Vector import Vector


class Base(object):
    """
    This superclass consists the radiuses, velocities
    and methods to work with them
     """

    def __init__(self, radiuses: Vector, velocities: Vector) -> None:
        self.radiuses = radiuses.to_dict()
        self.velocities = velocities.to_dict()
        self.number_of_particles = len(radiuses)
        self.basis = radiuses.get_keys()

    def radius_differences(self) -> dict:
        """
        This method realise the calculation of radius differences

        :return: Dictionary with radius differences (dict)
        """

        differences = {}
        ones_matrix = np.ones((1, self.number_of_particles))
        for axis in self.basis:
            differences[axis] = radius_differences(self.radiuses[axis], ones_matrix)
        return differences


@numba.njit(fastmath=True)
def radius_differences(radiuses: np.ndarray, ones_matrix: np.ndarray) -> np.ndarray:
    return radiuses @ ones_matrix - ones_matrix.T @ radiuses.T


if __name__ == '__main__':
    pass
