import re
import numpy as np
import matplotlib.pyplot as plt
from math import dist


def calculate_distance(candidate):
    """
    Calculate the distance the candidates route takes
    :param candidate: the candidate we want to evaluate
    :return: the distance of the route specified by the candidate
    """
    distance = 0
    for index, point in enumerate(candidate[:-1]):
        point1 = coordinates[point]
        point2 = coordinates[candidate[index+1]]
        distance += dist(point1, point2)
    return distance


def cross_helper(parent, offspring, start, end):
    """
    Helps crossover append "valid" cities picked from parent to the offspring
    i.e. cities not already in the offspring
    :param parent: parent we want to sample cities from
    :param offspring: current offspring
    :param start: starting index (for optimization purposes)
    :param end: end index
    :return: part of the offspring that needs to be appended to the current offspring
    """
    part_offspring = np.array([], dtype=np.int32)
    stop = False
    while start < end and not stop:
        counter = start
        while counter < len(parent):
            point = parent[counter]
            if point in offspring or point in part_offspring:
                counter += 1
                if counter == len(parent):
                    stop = True
            else:
                counter = len(parent)
                start += 1
                part_offspring = np.append(part_offspring, point)
    return part_offspring


def crossover(parent1, parent2):
    """
    Perform 2-point crossover with 2 candidates
    :param parent1: first parent for crossover
    :param parent2: second parent for crossover
    :return: offspring produced by performing 2-point crossover with the 2 parent
    """
    cp_left = np.random.randint(0, n_coordinates-1)
    cp_right = np.random.randint(cp_left+1, n_coordinates)

    offspring1 = parent1[cp_left:cp_right+1]
    left_part = cross_helper(parent2, offspring1, 0, cp_left)
    offspring1 = np.concatenate((left_part, offspring1))
    right_part = cross_helper(parent2, offspring1, cp_left, n_coordinates)
    offspring1 = np.concatenate((offspring1, right_part))

    offspring2 = parent2[cp_left:cp_right+1]
    left_part = cross_helper(parent1, offspring2, 0, cp_left)
    offspring2 = np.concatenate((left_part, offspring2))
    right_part = cross_helper(parent1, offspring2, cp_left, n_coordinates)
    offspring2 = np.concatenate((offspring2, right_part))

    return offspring1, offspring2


def rev_seq_mutation(sequence):
    """
    picks two points within sequence and swaps them
    :param sequence: the sequence to swap
    :return: sequence with the 2 selected cities swapped
    """
    swap_left = np.random.randint(0, n_coordinates - 1)
    swap_right = np.random.randint(swap_left + 1, n_coordinates)
    sequence[swap_left], sequence[swap_right] = sequence[swap_right], sequence[swap_left]
    return sequence


def opt_swap(route, j, k):
    """
    Swaps the route between points j and k according to the 2-opt algorithm
    :param route: current route
    :param j: begin index of cities to swap
    :param k: end index of cities to swap
    :return: modified route
    """
    return np.concatenate((route[:j], route[k:-len(route) + j - 1:-1], route[k + 1:len(route)]))


def opt_algorithm(candidate):
    """
    Perform 2-opt local search
    :param candidate: current candidate to be optimized
    :return: candidate optimized in one pass of the 2-opt algorithm
    """
    best_distance = calculate_distance(candidate)
    for v1 in range(len(candidate)-1):
        for v2 in range(v1+1, len(candidate)):
            new_candidate = opt_swap(candidate, v1, v2)
            new_distance = calculate_distance(new_candidate)
            if new_distance < best_distance:
                candidate = new_candidate
                best_distance = new_distance
    return candidate


def tournament_selection(candidate_pool, size):
    """
    Do binary tournament selection with replacement until we have n_selections candidates
    :param candidate_pool: the candidates which can be picked
    :param size: size of the tournament
    :return: candidates picked in the tournament selection
    """
    pairs = np.empty((n_selections, n_coordinates), np.int32)
    participant_pool = np.arange(0, pop_size)
    for select in range(n_selections):
        participants = np.random.choice(participant_pool, size, False)
        scores = np.array(list(map(calculate_distance, candidate_pool[participants])))
        scores_sorted = np.argsort(scores)
        winner = participants[scores_sorted[0]]
        pairs[select] = candidate_pool[winner]
    return pairs


def evolutionary_algorithm(cur_candidates, local_search):
    """
    Perform evolutionary algorithm to optimize a TSP problem
    :param cur_candidates: the current candidates in the population
    :param local_search: whether to apply local search
    :return: the best and average fitness of the candidates per iteration
    """
    best_distance = np.zeros(n_iterations)
    avg_distance = np.zeros(n_iterations)

    if local_search:
        cur_candidates = np.array(list(map(opt_algorithm, cur_candidates)))

    for iteration in range(n_iterations):
        # selection takes place here
        cross_pairs = tournament_selection(cur_candidates, 2)
        offspring = np.empty((n_selections, n_coordinates), np.int32)
        for pair_index in range(0, len(cross_pairs), 2):
            parent1 = cross_pairs[pair_index]
            parent2 = cross_pairs[pair_index+1]

            # crossover
            offspring1, offspring2 = crossover(parent1, parent2)

            # mutate
            if np.random.rand() < p_mutate:
                offspring1 = rev_seq_mutation(offspring1)
            if np.random.rand() < p_mutate:
                offspring2 = rev_seq_mutation(offspring2)

            offspring[pair_index] = offspring1
            offspring[pair_index+1] = offspring2

        all_candidates = np.concatenate((cur_candidates, offspring), axis=0)

        if local_search:
            all_candidates = np.array(list(map(opt_algorithm, all_candidates)))

        scores = np.array(list(map(calculate_distance, all_candidates)))
        scores_sorted = np.argsort(scores)
        cur_candidates = all_candidates[scores_sorted[:pop_size]]

        best_distance[iteration] = scores[scores_sorted[0]]
        avg_distance[iteration] = np.average(scores)
    return 1/best_distance, 1/avg_distance


# Plotting and handling running the functions with the correct file and settings
fig, ax = plt.subplots(2, 2, figsize=(12, 12))
titles = ['EA', 'MA']
files = ['file-tsp.txt', 'berlin52.txt']
p_mutate = 0.05
n_selections = 4
for instance in range(4):
    id_1 = instance // 2
    id_2 = instance % 2
    coordinate_file = files[id_1]
    coordinates = {}
    with open(coordinate_file) as file:
        for i, coordinate in enumerate(file):
            x, y = re.findall(r'\d+\.\d+', coordinate)
            coordinates[i] = (float(x), float(y))

    for run in range(10):
        n_coordinates = len(coordinates)
        pop_size = 6
        candidates = np.empty((pop_size, n_coordinates), np.int32)
        for i in range(pop_size):
            candidates[i] = np.random.permutation(list(coordinates.keys()))

        n_iterations = 1500
        fitness, average_fit = evolutionary_algorithm(candidates, id_2)
        print(f'{run+1} runs completed')
        ax[id_1, id_2].plot(np.arange(0, n_iterations), fitness, color='g')
        ax[id_1, id_2].plot(np.arange(0, n_iterations), average_fit, color='b', alpha=0.4)
        ax[id_1, id_2].set_xlabel('n_iterations')
        ax[id_1, id_2].set_ylabel('fitness')
        ax[id_1, id_2].set_title(f'Fitness of the {titles[id_2]} over iterations on the {files[id_1]} data')

colors = {'best': 'green', 'average': 'blue'}
labels = list(colors.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
fig.legend(handles, labels)
plt.show()
""" Calculate optimal path's distance for Berlin52.txt
opt = np.array([0, 48, 31, 44, 18, 40, 7, 8, 9, 42, 32, 50, 10, 51, 13, 12, 46, 25, 26, 27, 11, 24, 3, 5,
                14, 4, 23, 47, 37, 36, 39, 38, 35, 34, 33, 43, 45, 15, 28, 49, 19, 22, 29, 1, 6, 41, 20,
                16, 2, 17, 30, 21])
print(calculate_distance(opt))
"""
