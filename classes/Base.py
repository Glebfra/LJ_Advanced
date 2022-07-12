from classes.Vector import Vector


class Base(object):
    """This superclass consists the radiuses, velocities and methods to work with them"""

    def __init__(self, radiuses: Vector, velocities: Vector, mass: float) -> None:
        self.radiuses = radiuses
        self.velocities = velocities
        self.mass = mass
        self.number_of_particles = len(radiuses)
        self.basis = radiuses.get_keys()

    @property
    def radius_differences(self) -> Vector:
        """
        This method realise the calculation of radius differences

        :return: Dictionary with radius differences (dict)
        """

        differences = self.radiuses.differences()
        return differences


if __name__ == '__main__':
    pass
