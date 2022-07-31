from classes.Base import Base
from classes.Vector import Vector


class LJ(Base):
    """This class describes the LJ particular interactions"""

    def __init__(self, radiuses: Vector, velocities: Vector, sigma: float, eps: float, mass: float) -> None:
        super().__init__(radiuses, velocities, mass)

        self.sigma = sigma
        self.eps = eps

    @property
    def potential(self) -> float:
        """
        Potential energy of LJ system
        """

        differences = self.radius_differences
        r = abs(differences)
        temp = self.sigma / r
        return 4 * self.eps * (temp ** 12 - temp ** 6).sum()

    @property
    def force(self) -> Vector:
        """
        Forces of LJ system
        """

        differences = self.radius_differences
        r = abs(differences)
        temp = self.sigma / r
        forces = differences * (8 * self.eps * (12 * temp ** 14 / self.sigma ** 2 - 6 * temp ** 8 / self.sigma ** 2))
        return forces.sum_columns()

    @property
    def acceleration(self) -> Vector:
        """Accelerations of LJ system"""
        return self.force / self.mass

    @property
    def kinetic(self) -> float:
        """Kinetic energy of LJ System"""
        return 0.5 * self.mass * (self.velocities ** 2).sum()

    @property
    def hamilton(self) -> float:
        """Hamiltonian of LJ System"""
        return self.potential + self.kinetic
