import numpy as np
from classes.LJ import LJ
from classes.Vector import Vector


class System(LJ):
    """This class describes the system"""

    def __init__(self, radiuses: Vector, velocities: Vector, sigma: float, eps: float,
                 temperature: float, mass: float, cube_length) -> None:
        super().__init__(radiuses, velocities, sigma, eps)

        self.temperature = temperature
        self.mass = mass
        self.momentum_temperature = 0
        self.cube_length = cube_length

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
            'sigma': 1e-10,
            'eps': 120 * boltsman,
            'temperature': temperature,
            'mass': mass,
            'cube_length': cube_length
        }
        return cls(**properties)

    def next_time_turn(self, delta_time: float) -> None:
        force = self.force()
        self.radiuses += self.velocities * delta_time + (force / self.mass * delta_time ** 2) / 2
        self.velocities = (self.velocities * self.get_velocity_coef()) + force / self.mass * delta_time
        self.periodic_boundary_conditions()

    def get_velocity_coef(self) -> float:
        self.momentum_temperature = 0
        return 1

    def periodic_boundary_conditions(self):
        self.radiuses -= (self.radiuses // self.cube_length) * self.cube_length


if __name__ == '__main__':
    system = System.create_default_2D_system(number_of_particles=5, cube_length=1e-8, temperature=300)
    print(system.radiuses)
