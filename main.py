from biology import GeneticAlg, NN
from Snake_Game import *

# Important to remember:
#  - Get initial population
#  - Let nature (but not really...) do it's job
#  - ???
#  - Hopefully profit

# humans have 23 pairs, right? so total of 46 chromosomes? dont know, I am gonna start with that
nchromosomes = 46
nweights = NN.in_layer * NN.hidden1 + NN.hidden1 * NN.hidden2 + NN.hidden2 * NN.out_layer

# population and lifespan
p_size = (nchromosomes, nweights)
population = GeneticAlg.generate_random_population(p_size)
ngenerations = 100
n_parents_mating = 6  # it should be an even number!

# Order to remember for the nature, as we already have a random population selected before the nature run:
#  - Get the fitness
#  - Natural selection, survival of the fittest, or some nazi experiment, no clue (SELECTION)
#  - Breed them (CROSSOVER)
#  - Send them to Chernobyl (MUTATION)
#  - Your new population now has the fittest parents, and their kids, the weaks died
#  - Rinse and repeat until number of generations has passed

for gen in range(ngenerations):
    print("Current generation: " + str(gen))

    # Fitness
    fitness = GeneticAlg.calc_fitness(population)
    print("Fittest chromosome value: " + str(np.max(fitness)))

    # Get the best parents and breed them
    parents = GeneticAlg.natural_selection(population, fitness, n_parents_mating)
    kids = GeneticAlg.breeding(parents, size=(p_size[0] - parents.shape[0], nweights))

    # Mutations
    mutant_kids = GeneticAlg.chernobyl(kids)

    # Population Update
    population[0:parents.shape[0], :] = parents
    population[parents.shape[0]:, :] = mutant_kids

