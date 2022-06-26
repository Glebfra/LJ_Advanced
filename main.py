import numpy as np

from classes.System import System
import matplotlib.pyplot as plt


def main():
    # Creating the system
    np.random.seed(1)
    system = System.create_default_3D_system(number_of_particles=200, cube_length=1e-8, temperature=300)

    evolution_time = 1e-10
    number_of_iterations = 1000
    delta_time = evolution_time / number_of_iterations
    H = []

    # Time loop
    for i in range(number_of_iterations):
        system.next_time_turn(delta_time)
        H.append(system.hamilton)

    time = np.linspace(0, evolution_time, number_of_iterations)
    plt.figure(1)
    plt.plot(time, H, 'r-')
    plt.show()

    avg_H = sum(H) / len(H)
    error = (max(H) - avg_H) / avg_H
    print(f'Error: {round(error * 100, 3)} %')


if __name__ == '__main__':
    main()
