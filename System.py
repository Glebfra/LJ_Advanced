import numpy as np
from LJ import LJ


class System(object):
    def __init__(self, radiuses: dict, velocities: dict, mass: float, temperature: float, cube_length: float) -> None:
        self.radiuses = radiuses
        self.velocities = velocities
        self.mass = mass
        self.temperature = temperature
        self.cube_length = cube_length
        self.number_of_particles = len(self.radiuses['x'])

        self.boltsman = 1.38e-23

        self.lj = LJ.create_default_LJ()

    @classmethod
    def create_default_system(cls, number_of_particles, cube_length, temperature):
        radiuses, velocities = {}, {}
        boltsman, mass = 1.38e-23, 1.6e-27
        start_velocity = np.sqrt(boltsman * temperature / mass)
        for axis in ['x', 'y', 'z']:
            radiuses[axis] = np.random.sample((number_of_particles, 1)) * cube_length
            velocities[axis] = (2 * np.random.sample((number_of_particles, 1)) - 1) * start_velocity

        return cls(radiuses=radiuses, velocities=velocities, mass=mass, temperature=temperature,
                   cube_length=cube_length)

    def radius_differences(self) -> dict:
        differences = {}
        E = np.ones((1, self.number_of_particles))
        for axis in self.radiuses:
            temp = self.radiuses[axis]
            differences[axis] = np.dot(temp, E) - np.dot(E.T, temp.T)
        return differences

    def calculate_velocity_coef(self):
        temp = 0
        for axis in self.velocities:
            temp += self.velocities[axis] ** 2
        momentum_temperature = 2 / (3 * self.number_of_particles * self.boltsman) * (self.mass * temp.sum())
        return np.sqrt(self.temperature / momentum_temperature)

    def periodic_boundary_conditions(self):
        for axis in self.radiuses:
            temp = self.radiuses[axis] // self.cube_length
            self.radiuses[axis] -= temp * self.cube_length

    def next_time_turn(self, delta_time: float) -> None:
        acceleration = self.lj.force(self.radius_differences()) / self.mass
        velocity_coef = self.calculate_velocity_coef()
        for axis in self.radiuses:
            self.radiuses[axis] += self.velocities[axis] * delta_time + (acceleration[axis] * delta_time ** 2) / 2
            self.velocities[axis] = velocity_coef * self.velocities[axis] + acceleration[axis] * delta_time
        self.periodic_boundary_conditions()


if __name__ == '__main__':
    system = System.create_default_system(number_of_particles=10, cube_length=1e-8, temperature=300)

