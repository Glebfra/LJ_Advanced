import numpy as np
from classes.LJ import LJ
from classes.Vector import Vector


class System(LJ):
    """This class describes the system"""

    def __init__(self, radiuses: Vector, velocities: Vector, sigma: float, eps: float,
                 temperature: float, mass: float, cube_length) -> None:
        super().__init__(radiuses, velocities, sigma, eps, mass)

        self.temperature = temperature
        self.momentum_temperature = temperature
        self.cube_length = cube_length
        self.boltsman = 1.38e-23

    @classmethod
    def create_default_2D_system(cls, number_of_particles: int, cube_length: float, temperature: float):
        boltsman, mass = 1.38e-23, 1.6e-27
        start_velocity = np.sqrt(boltsman * temperature / mass)
        properties = {
            'radiuses': Vector({axis: np.random.sample((number_of_particles, 1)) * cube_length for axis in 'xy'}),
            'velocities': Vector(
                {axis: (2 * np.random.sample((number_of_particles, 1)) - 1) * start_velocity for axis in 'xy'}),
            'sigma': 51e-12,
            'eps': 120 * boltsman,
            'temperature': temperature,
            'mass': mass,
            'cube_length': cube_length
        }
        return cls(**properties)

    @classmethod
    def create_default_3D_system(cls, number_of_particles: int, cube_length: float, temperature: float):
        boltsman, mass = 1.38e-23, 1.6e-27
        start_velocity = np.sqrt(boltsman * temperature / mass)
        properties = {
            'radiuses': Vector({axis: np.random.sample((number_of_particles, 1)) * cube_length for axis in 'xyz'}),
            'velocities': Vector(
                {axis: (2 * np.random.sample((number_of_particles, 1)) - 1) * start_velocity for axis in 'xyz'}),
            'sigma': 51e-12,
            'eps': 120 * boltsman,
            'temperature': temperature,
            'mass': mass,
            'cube_length': cube_length
        }
        return cls(**properties)

    def next_time_turn(self, delta_time: float) -> None:
        acceleration_old = self.acceleration

        self.radiuses += self.velocities * delta_time + 0.5 * acceleration_old * delta_time ** 2
        acceleration_new = self.acceleration

        self.velocities *= self.velocity_coef
        self.velocities += 0.5 * (acceleration_old + acceleration_new) * delta_time

        self.periodic_boundary_conditions()

    @property
    def velocity_coef(self) -> Vector:
        self.momentum_temperature = (self.velocities ** 2) * self.mass / (
                3 * self.boltsman * self.number_of_particles)
        return (self.temperature / self.momentum_temperature).sum() ** (1 / 2)

    def periodic_boundary_conditions(self):
        self.radiuses = self.radiuses - (self.radiuses // self.cube_length) * self.cube_length


if __name__ == '__main__':
    pass
