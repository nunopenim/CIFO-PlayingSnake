# game imports
from Snake_Game import *

# biology imports (implementation1)
from biology import GeneticAlg, NN

# charles imports
from charles.charles import Individual, Population
from charles.crossover import single_point_co, arithmetic_co, pmx_co, cycle_co
from charles.mutation import inversion_mutation
from charles.selection import fps, tournament

# snake imports (implementation2)
from snake import get_fitness, get_neighbours


# Important to remember:
#  - Get initial population
#  - Let nature (but not really...) do it's job
#  - ???
#  - Hopefully profit

# Order to remember for the nature, as we already have a random population selected before the nature run:
#  - Get the fitness
#  - Natural selection, survival of the fittest, or some nazi experiment, no clue (SELECTION)
#  - Breed them (CROSSOVER)
#  - Send them to Chernobyl (MUTATION)
#  - Your new population now has the fittest parents, and their kids, the weaks died
#  - Rinse and repeat until number of generations has passed


# first version of the snake implementation
def implementation1():

    # humans have 23 pairs, right? so total of 46 chromosomes? dont know, I am gonna start with that
    nchromosomes = 46
    nweights = NN.in_layer * NN.hidden1 + NN.hidden1 * NN.hidden2 + NN.hidden2 * NN.out_layer

    # population and lifespan
    p_size = (nchromosomes, nweights)
    population = GeneticAlg.generate_random_population(p_size)
    ngenerations = 200
    n_parents_mating = 6  # it should be an even number!

    for gen in range(ngenerations):
        print("Current generation: " + str(gen))

        # Fitness
        fitness = GeneticAlg.calc_fitness(population)
        print("Fittest chromosome value: " + str(np.max(fitness)))

        # Selection
        parents = GeneticAlg.natural_selection(population, fitness, n_parents_mating)

        # Crossover
        kids = GeneticAlg.breeding(parents, size=(p_size[0] - parents.shape[0], nweights))

        # Mutation
        mutant_kids = GeneticAlg.chernobyl(kids)

        # Population Update
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutant_kids


def implementation2():
    nchromosomes = 200
    nweights = NN.in_layer * NN.hidden1 + NN.hidden1 * NN.hidden2 + NN.hidden2 * NN.out_layer

    # population and lifespan

    # Monkey Patching
    Individual.get_fitness = get_fitness
    Individual.get_neighbours = get_neighbours

    pop = Population(
        size=nchromosomes, optim="max", sol_size=nweights, valid_set=(np.arange(-1, 1.001, step=0.001)).tolist(), replacement=True
    )

    pop.evolve(gens=15000,
               select=fps,
               crossover=single_point_co,
               mutate=inversion_mutation,
               co_p=0.9, mu_p=0.1,
               elitism=True)


# UNCOMMENT WHICH VERSION YOU WANT TO RUN
#implementation1()
implementation2()
