import numpy as np

from classes.GpuVector import GpuVector
from classes.LJ import LJ
from classes.Vector import Vector


class System(LJ):
    """This class describes the system founded on Lenard Jones particular interactions"""

    def __init__(self, radiuses: Vector, velocities: Vector, sigma: float, eps: float,
                 temperature: float, mass: float, cube_length) -> None:
        super().__init__(radiuses, velocities, sigma, eps, mass)

        self.temperature = temperature
        self.momentum_temperature = temperature
        self.cube_length = cube_length
        self.boltsman = 1.38e-23

    @classmethod
    def create_default_2D_system(cls, number_of_particles: int, cube_length: float, temperature: float):
        def _particles_overlap(radiuses: Vector, sigma: float) -> Vector:
            differences = radiuses.differences()
            r = abs(differences)
            if r.min() < sigma * 1.1:
                print(f'Now the coordinates are configuring')
                new_radiuses = Vector(
                    {axis: np.random.sample((number_of_particles, 1), dtype=np.float32) * cube_length for axis in 'xy'})
                _particles_overlap(new_radiuses, sigma)
            else:
                return radiuses

        boltsman, mass = 1.38e-23, 6.69e-26
        start_velocity = np.sqrt(boltsman * temperature / mass)
        properties = {
            'radiuses': Vector({axis: np.random.sample((number_of_particles, 1)) * cube_length for axis in 'xy'}),
            'velocities': Vector(
                {axis: (2 * np.random.sample((number_of_particles, 1)) - 1) * start_velocity for axis in 'xy'}),
            'sigma': 3.4e-10,
            'eps': 119.8 * boltsman,
            'temperature': temperature,
            'mass': mass,
            'cube_length': cube_length
        }
        properties['radiuses'] = _particles_overlap(properties['radiuses'], properties['sigma'])
        return cls(**properties)

    @classmethod
    def create_default_3D_system(cls, number_of_particles: int, cube_length: float, temperature: float):
        def _particles_overlap(radiuses: Vector, sigma: float) -> Vector:
            print(f'Now the coordinates are configuring')
            differences = radiuses.differences()
            r = abs(differences)
            if r.min() < sigma * 1.1:
                new_radiuses = Vector(
                    {axis: np.random.sample((number_of_particles, 1)) * cube_length for axis in 'xyz'})
                _particles_overlap(new_radiuses, sigma)
            else:
                return radiuses

        boltsman, mass = 1.38e-23, 6.69e-26
        start_velocity = np.sqrt(boltsman * temperature / mass)
        properties = {
            'radiuses': Vector(
                {axis: np.float32(np.random.sample((number_of_particles, 1))) * cube_length for axis in 'xyz'}),
            'velocities': GpuVector.create_vector_from_dict(
                {axis: np.float32((2 * np.random.sample((number_of_particles, 1)) - 1)) * start_velocity for axis in
                 'xyz'}),
            'sigma': 3.4e-10,
            'eps': 119.8 * boltsman,
            'temperature': temperature,
            'mass': mass,
            'cube_length': cube_length
        }
        properties['radiuses'] = _particles_overlap(properties['radiuses'], properties['sigma'])
        properties['radiuses'] = GpuVector.create_vector_from_dict(properties['radiuses'].vector)

        return cls(**properties)

    def next_time_turn(self, delta_time: float) -> None:
        self.velocities = self.velocities * self.velocity_coef
        self.velocities = self.velocities + self.acceleration * delta_time
        self.radiuses = self.radiuses + self.velocities * delta_time

        self.periodic_boundary_conditions()

    @property
    def velocity_coef(self) -> float:
        self.momentum_temperature = (self.velocities ** 2).sum() * self.mass / (
                3 * self.boltsman * self.number_of_particles)
        return (self.temperature / self.momentum_temperature) ** (1 / 2)

    def periodic_boundary_conditions(self) -> None:
        self.radiuses -= (self.radiuses // self.cube_length) * self.cube_length

    def boundary_conditions(self) -> None:
        pass


if __name__ == '__main__':
    pass
