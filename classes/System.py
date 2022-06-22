import numpy as np
from LJ import LJ
from Vector import Vector


class System(LJ):
    """This class describes the system"""

    def __init__(self, radiuses: Vector, velocities: Vector, sigma: float, eps: float,
                 temperature: float, mass: float) -> None:
        super().__init__(radiuses, velocities, sigma, eps)

        self.temperature = temperature
        self.mass = mass
        self.momentum_temperature = 0

    @classmethod
    def create_default_2D_system(cls, number_of_particles: int, cube_length: float, temperature: float):
        boltsman, mass = 1.38e-23, 1.6e-27
        start_velocity = np.sqrt(boltsman * temperature / mass)
        properties = {
            'radiuses': Vector({
                'x': np.random.sample((number_of_particles, 1)) * cube_length,
                'y': np.random.sample((number_of_particles, 1)) * cube_length
            }),
            'velocities': Vector({
                'x': (2 * np.random.sample((number_of_particles, 1)) - 1) * start_velocity,
                'y': (2 * np.random.sample((number_of_particles, 1)) - 1) * start_velocity
            }),
            'sigma': 51e-12,
            'eps': 120 * boltsman,
            'temperature': temperature,
            'mass': mass
        }
        return cls(**properties)

    @classmethod
    def create_default_3D_system(cls, number_of_particles: int, cube_length: float, temperature: float):
        boltsman, mass = 1.38e-23, 1.6e-27
        start_velocity = np.sqrt(boltsman * temperature / mass)
        properties = {
            'radiuses': Vector({
                'x': np.random.sample((number_of_particles, 1)) * cube_length,
                'y': np.random.sample((number_of_particles, 1)) * cube_length,
                'z': np.random.sample((number_of_particles, 1)) * cube_length
            }),
            'velocities': Vector({
                'x': (2 * np.random.sample((number_of_particles, 1)) - 1) * start_velocity,
                'y': (2 * np.random.sample((number_of_particles, 1)) - 1) * start_velocity,
                'z': (2 * np.random.sample((number_of_particles, 1)) - 1) * start_velocity
            }),
            'sigma': 1e-10,
            'eps': 120 * boltsman,
            'temperature': temperature,
            'mass': mass
        }
        return cls(**properties)

    def next_time_turn(self, delta_time: float) -> None:
        force = self.force()
        velocity_coef = self.get_velocity_coef()
        self.radiuses += self.velocities * delta_time + (force / self.mass * delta_time ** 2) / 2
        self.velocities = (self.velocities * velocity_coef) + force / self.mass * delta_time

    def get_velocity_coef(self) -> float:
        pass


if __name__ == '__main__':
    system = System.create_default_2D_system(number_of_particles=5, cube_length=1e-7, temperature=300)
    print(system.force().to_dict())
