import numpy as np
from random import randint

# Game Stuff
from Snake_Game import display, clock
from Snake_Game import starting_positions, blocked_directions, angle_with_apple, generate_button_direction, \
    collision_with_boundaries, collision_with_self, play_game

# This is for crossover tuning adjustments, higher means a finer selection, lower means coarser
BREEDING_TUNING_VALUE = 100

# Auxiliary stuff
VERY_SMALL_NUMBER = -2147483646  # This has to be a C Long for numpy to be happy, otherwise Overflow!


class NN:
    in_layer = 7  # the in-layer needs to be 7 because of 7 parameters in
    hidden1 = 32  # this one is double the other for no specific reason
    hidden2 = 64  # same as previous
    out_layer = 3  # just a smaller number I guess

    weight1_shape = (hidden1, in_layer)
    weight2_shape = (hidden2, hidden1)
    weight3_shape = (out_layer, hidden2)

    # Auxiliary
    @staticmethod
    def softmax(fz):
        return np.exp(fz.T) / np.sum(np.exp(fz.T), axis=1).reshape(-1, 1)

    @staticmethod
    def weights(i):
        weight1 = i[0:NN.weight1_shape[0] * NN.weight1_shape[1]]
        weight2 = i[NN.weight1_shape[0] * NN.weight1_shape[1]:NN.weight2_shape[0] * NN.weight2_shape[1] +
                                                              NN.weight1_shape[0] * NN.weight1_shape[1]]
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

    # Generates a random population, used on the start of each run
    @staticmethod
    def generate_random_population(size):
        # population = random.choice(np.arange(-1, 1, step=0.01))  # you donut, this needs to be with np.random!
        population = np.random.choice(np.arange(-1, 1, step=0.01), size=size, replace=True)
        return np.array(population)

    # Compute Fitness
    @staticmethod
    def calc_fitness(population):
        fitnesses = []
        for i in range(population.shape[0]):
            fitness = machine_play(display, clock, population[i])
            # print('chromosome: ' + str(i) + ", fitness: " + str(fitness))
            fitnesses.append(fitness)
        return np.array(fitnesses)

    # Selection
    @staticmethod
    def natural_selection(population, fitness, nparents):
        parents = np.empty((nparents, population.shape[1]))

        # basically we choose the best nparents from our population and return an numpy array of them
        for parent_num in range(nparents):
            max_fitness_idx = np.where(fitness == np.max(fitness))[0][0]
            parents[parent_num, :] = population[max_fitness_idx, :]
            fitness[max_fitness_idx] = VERY_SMALL_NUMBER
        return parents

    # Crossover
    @staticmethod
    def breeding(parents, size):
        offspring_size = size[0]
        weights = size[1]
        offspring = np.empty(size)
        for k in range(offspring_size):

            # We get 2 random different parents
            parent1 = randint(0, parents.shape[0] - 1)
            parent2 = randint(0, parents.shape[0] - 1)
            # This is a safeguard so that they don't breed with themselves, because... duh?
            while parent1 == parent2:
                parent1 = randint(0, parents.shape[0] - 1)
                parent2 = randint(0, parents.shape[0] - 1)

            # We make the crossover, randomly, based on RNG
            for j in range(weights):
                # Basically, here I am choosing to get a random number with equal probabilities, then if lower
                # than half, we choose the chromosome from the first parent, otherwise the second parent.
                if randint(0, BREEDING_TUNING_VALUE) < BREEDING_TUNING_VALUE / 2:
                    offspring[k, j] = parents[parent1, j]
                else:
                    offspring[k, j] = parents[parent2, j]

        return offspring

    # Mutation - this still needs a bit of work
    @staticmethod
    def chernobyl(offspring):
        mutated_offspring = offspring.copy()

        # We select a random individual from the pool, then a random chromosome, and mutate it
        for idx in range(offspring.shape[0]):

            # 50/50 chance of mutation
            if randint(0, 100) < 50:
                continue

            # Mutation
            rand = randint(0, offspring.shape[1] - 1)
            random_value = np.random.choice(np.arange(-1, 1, step=0.001), size=1, replace=False)
            mutated_offspring[idx, rand] = offspring[idx, rand] + random_value
        return mutated_offspring


# Intelligence and fitness function
def machine_play(display, clock, weights):
    weights = np.array(weights)
    max_score = 0
    avg_score = 0
    test_games = 1
    frame_score = 0
    steps_per_game = 2500
    noApple = 0

    for _ in range(test_games):

        snake_start, snake_position, apple_position, score = starting_positions()

        count_same_direction = 0
        prev_direction = 0


        for _ in range(steps_per_game):
            current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(
                snake_position)
            angle, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized = angle_with_apple(
                snake_position, apple_position)
            predictions = []
            predicted_direction = np.argmax(np.array(NN.forward_propagation(np.array(
                [is_left_blocked, is_front_blocked, is_right_blocked, apple_direction_vector_normalized[0],
                 snake_direction_vector_normalized[0], apple_direction_vector_normalized[1],
                 snake_direction_vector_normalized[1]]).reshape(-1, 7), weights))) - 1

            if predicted_direction == prev_direction:
                count_same_direction += 1
            else:
                count_same_direction = 0
                prev_direction = predicted_direction

            new_direction = np.array(snake_position[0]) - np.array(snake_position[1])
            if predicted_direction == -1:
                new_direction = np.array([new_direction[1], -new_direction[0]])
            if predicted_direction == 1:
                new_direction = np.array([-new_direction[1], new_direction[0]])

            button_direction = generate_button_direction(new_direction)

            next_step = snake_position[0] + current_direction_vector
            if collision_with_boundaries(snake_position[0]) == 1 or collision_with_self(next_step.tolist(),
                                                                                        snake_position) == 1:
                break

            else:
                frame_score += 1

            snake_position, apple_position, score = play_game(snake_start, snake_position, apple_position,
                                                              button_direction, score, display, clock)

            if score > max_score:
                max_score = score
                noApple = 0
            else:
                noApple += 1

            # STOP IF NO APPLE FOR * TURNS
            if noApple >= 150:
                break

            if count_same_direction > 8 and predicted_direction != 0:
                pass
            else:
                pass

    # Subtract the number of frames since the last fruit was eaten from the fitness
    # This is to discourage snakes from trying to gain fitness by avoiding fruit
    if noApple >= 150:
        frame_score = frame_score - noApple

    # Ensure we do not multiply fitness by a factor of 0
    if frame_score <= 0:
        frame_score = 1

    # finalScore = score1 + score2 + max_score * 10
    finalScore = (max_score * 2) ** 2 * (frame_score ** 0.15)

    if finalScore < 0:
        return 1
    else:
        return finalScore
