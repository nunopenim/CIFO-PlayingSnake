from copy import deepcopy
import random

import biology
from Snake_Game import display, clock

from charles.charles import Individual


# Compute Fitness
def get_fitness(self):
    """A fitness function for the snake game.
    """
    fitness = biology.machine_play(display, clock, self.representation)
    return fitness


def get_neighbours(self):
    """A neighbourhood function for the snake game.
    """
    n = [deepcopy(self.representation) for i in range(len(self.representation))]

    for count, i in enumerate(n):
        if i[count] == -1:
            i[count] = -0.99
        elif i[count] == 0.99:
            i[count] = 0.98
        else:
            roll = random.randint(1, 100)

            if roll <= 50:
                i[count] = i[count] - 0.01
            elif roll >= 51:
                i[count] = i[count] + 0.01

    n = [Individual(i) for i in n]
    return n
