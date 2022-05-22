import numpy as np
from play_the_game import machine_play
from Snake_Game import display, clock
from random import choice, randint, uniform

class NN:
    in_layer = 5
    hidden1 = 25
    hidden2 = 10
    out_layer = 1

    weight1_shape = (25, 5)
    weight2_shape = (10, 25)
    weight3_shape = (1, 10)


    # Auxiliary
    @staticmethod
    def softmax(fz):
        return np.exp(fz.T) / np.sum(np.exp(fz.T), axis=1).reshape(-1, 1)

    @staticmethod
    def weights(i):
        weight1 = i[0:NN.weight1_shape[0] * NN.weight1_shape[1]]
        weight2 = i[NN.weight1_shape[0] * NN.weight1_shape[1]:NN.weight2_shape[0] * NN.weight2_shape[1] + NN.weight1_shape[0] * NN.weight1_shape[1]]
        weight3 = i[NN.weight2_shape[0] * NN.weight2_shape[1] + NN.weight1_shape[0] * NN.weight1_shape[1]:]

        return \
            weight1.reshape(NN.weight1_shape[0], NN.weight1_shape[1]), \
            weight2.reshape(NN.weight2_shape[0], NN.weight2_shape[1]), \
            weight3.reshape(NN.weight3_shape[0], NN.weight3_shape[1])

    # The Network - tanh is probably better as the negatives are mapped to strong negatives!
    @staticmethod
    def forward_propagation(X, i):
        w1, w2, w3 = NN.weights(i)

        fz1 = np.matmul(w1, X.T)
        activation1 = np.tanh(fz1)
        fz2 = np.matmul(w2, activation1)
        activation2 = np.tanh(fz2)
        fz3 = np.matmul(w3, activation2)
        activation3 = NN.softmax(fz3)

        return activation3

class GeneticAlg:

    # Compute Fitness
    @staticmethod
    def calc_fitness(population):
        fitnesses = []
        for i in range(population.shape[0]):
            fitness = machine_play(display, clock, population[i])
            print('Chromossome number ' + str(i) + "fitness: " + str(fitness))
            fitnesses.append(fitness)
        return np.array(fitnesses)

    # Selection
    @staticmethod
    def natural_selection(pop, fitness, nparents):
        parents = np.empty((nparents, pop.shape[1]))
        for parent_num in range(nparents):
            max_fitness_idx = np.where(fitness == np.max(fitness))[0][0]
            parents[parent_num, :] = pop[max_fitness_idx, :]
            fitness[max_fitness_idx] = int(float('-Inf'))
        return np.array(parents)

    # Crossover
    @staticmethod
    def breeding(parents, size):
        offspring = np.empty(size)
        for k in range(size[0]):
            while True:
                parent1_idx = randint(0, parents.shape[0] - 1)
                parent2_idx = randint(0, parents.shape[0] - 1)
                # produce offspring from two parents if they are different
                if parent1_idx != parent2_idx:
                    for j in range(size[1]):
                        if uniform(0, 1) < 0.5:
                            offspring[k, j] = parents[parent1_idx, j]
                        else:
                            offspring[k, j] = parents[parent2_idx, j]
                    break
        return np.array(offspring)

    # Mutation
    @staticmethod
    def chernobyl(offspring):
        mutated_offspring = offspring.copy()
        for idx in range(offspring.shape[0]):
            for _ in range(25):
                i = randint(0, offspring.shape[1] - 1)
            random_value = np.random.choice(np.arange(-1, 1, step=0.001), size=1, replace=False)
            mutated_offspring[idx, i] = offspring[idx, i] + random_value
        return mutated_offspring

