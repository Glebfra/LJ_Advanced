import numpy as np
from Base import Base


class LJ(Base):
    """This class describes the LJ particular interactions"""

    def __init__(self, radiuses: dict, velocities: dict, sigma: float, eps: float) -> None:
        super().__init__(radiuses, velocities)
        self.sigma = sigma
        self.eps = eps

    def _calculate_radius(self, differences: dict) -> np.ndarray:
        r = 0
        for axis in differences:
            r += differences[axis] ** 2
        r = np.sqrt(r)
        r += np.eye(self.number_of_particles)
        return r

    def potential(self) -> float:
        """
        This method realise the calculation of potential energy

        :return: Potential energy (float)
        """

        r = self._calculate_radius(self.radius_differences())
        temp = self.sigma / r
        return 4 * self.eps * (temp ** 12 - temp ** 6).sum()

    def force(self) -> dict:
        """
        This method realise the calculation of force

        :return: Dictionary with forces (dict)
        """

        forces = {}
        differences = self.radius_differences()
        r = self._calculate_radius(differences)
        temp = self.sigma / r
        for axis in self.radiuses:
            forces[axis] = (8 * self.eps * (12 * temp ** 14 / self.sigma ** 2 - 6 * temp ** 8 / self.sigma ** 2) *
                            differences[axis]).sum(1).reshape((self.number_of_particles, 1))
        return forces
