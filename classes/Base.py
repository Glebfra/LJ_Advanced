import numpy as np
import numba


class Base(object):
    """
    This superclass consists the radiuses, velocities
    and methods to work with them
     """

    def __init__(self, radiuses: dict, velocities: dict) -> None:
        self.radiuses = radiuses
        self.velocities = velocities
        self.number_of_particles = len(self.radiuses['x'])

    def radius_differences(self) -> dict:
        """
        This method realise the calculation of radius differences

        :return: Dictionary with radius differences (dict)
        """

        differences = {}
        ones_matrix = np.ones((1, self.number_of_particles))
        for axis in self.radiuses:
            differences[axis] = radius_differences(self.radiuses[axis], ones_matrix)
        return differences


@numba.njit(fastmath=True)
def radius_differences(radiuses: np.ndarray, ones_matrix: np.ndarray) -> np.ndarray:
    return radiuses @ ones_matrix - ones_matrix.T @ radiuses.T
