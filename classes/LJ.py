from classes.Base import Base
from classes.Vector import Vector


class LJ(Base):
    """This class describes the LJ particular interactions"""

    def __init__(self, radiuses: Vector, velocities: Vector, sigma: float, eps: float) -> None:
        super().__init__(radiuses, velocities)

        self.sigma = sigma
        self.eps = eps

    def potential(self) -> float:
        """
        This method realise the calculation of potential energy

        :return: Potential energy (float)
        """

        differences = Vector(self.radius_differences())
        r = abs(differences)
        temp = self.sigma / r
        return 4 * self.eps * (temp ** 12 - temp ** 6).sum()

    def force(self) -> Vector:
        """
        This method realise the calculation of force

        :return: Vector with forces (Vector)
        """

        differences = Vector(self.radius_differences())
        r = abs(differences)
        temp = self.sigma / r
        forces = differences * (8 * self.eps * (12 * temp ** 14 / self.sigma ** 2 - 6 * temp ** 8 / self.sigma ** 2))
        return forces.sum_columns()
