import math
import operator
import random as rnd
import matplotlib.pyplot as plt

import numpy as np
from deap import gp, creator, base, tools
from deap.gp import genFull, PrimitiveTree

# Target function values
data = dict()
data[-1] = 0
data[-0.9] = -0.1629
data[-0.8] = -0.2624
data[-0.7] = -0.3129
data[-0.6] = -0.3264
data[-0.5] = -0.3125
data[-0.4] = -0.2784
data[-0.3] = -0.2289
data[-0.2] = -0.1664
data[-0.1] = -0.0909
data[0.0] = 0.0
data[0.1] = 0.1111
data[0.2] = 0.2496
data[0.3] = 0.4251
data[0.4] = 0.6496
data[0.5] = 0.9375
data[0.6] = 1.3056
data[0.7] = 1.7731
data[0.8] = 2.3616
data[0.9] = 3.0951
data[1.0] = 4.0000


def evaluate(expr: PrimitiveTree):
    expr_func = gp.compile(expr, pset)
    cost = 0
    try:
        for x in data.keys():
            cost += abs(expr_func(x) - data[x])
    except OverflowError:
        return math.nan,
    except ValueError:
        return math.nan,
    except ZeroDivisionError:
        return math.nan,

    # Bloat control
    cost += expr.height

    return -cost,


def initIterate(container, generator):
    ind = tools.initIterate(container, generator)
    while math.isnan(evaluate(ind)[0]):
        ind = tools.initIterate(container, generator)
    return ind


def recombination(ind1, ind2):
    c1, c2 = toolbox.clone(ind1), toolbox.clone(ind2)
    gp.cxOnePointLeafBiased(c1, c2, 0.1)
    while math.isnan(evaluate(c1)[0]) or math.isnan(evaluate(c2)[0] == math.nan):
        c1, c2 = toolbox.clone(ind1), toolbox.clone(ind2)
        gp.cxOnePointLeafBiased(c1, c2, 0.1)
    return c1, c2


def average_fitness(population) -> float:
    sum_fitness = 0
    for p in population:
        sum_fitness += evaluate(p)[0]
    return sum_fitness / len(population)


def average_node(population):
    sum_size = 0
    for p in population:
        sum_size += len(p)
    return sum_size / len(population)


def get_best_ind(population):
    best_ind = population[0]
    best_ind_fit = evaluate(population[0])
    for ind in population[1:]:
        fit = evaluate(ind)
        if fit > best_ind_fit:
            best_ind = ind
            best_ind_fit = fit

    return toolbox.clone(best_ind), best_ind_fit


def get_max_node(population):
    max_ind = population[0]
    max_ind_size = len(population[0])
    for ind in population[1:]:
        size = len(ind)
        if size > max_ind_size:
            max_ind = ind
            max_ind_size = size
    return max_ind, max_ind_size


def get_min_node(population):
    min_ind = population[0]
    min_ind_size = len(population[0])
    for ind in population[1:]:
        size = len(ind)
        if size < min_ind_size:
            min_ind = ind
            min_ind_size = size
    return min_ind, min_ind_size


# Function set definition
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.truediv, 2)
pset.addPrimitive(math.log, 1)
pset.addPrimitive(math.exp, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)
pset.renameArguments(ARG0="x")

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=10)
toolbox.register("individual", initIterate, creator.Individual, toolbox.expr)
toolbox.register("recombination", recombination)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("evaluate", evaluate)


def gen_symb_expr(n, gen, cx_prob):
    gp.staticLimit(operator.attrgetter('height'), 17)
    population = [toolbox.individual() for _ in range(n)]
    best_ind = [get_best_ind(population)[0]]
    all_ind = [population]

    fitnesses = [toolbox.evaluate(ind) for ind in population]
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for i in range(gen):
        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = [toolbox.clone(p) for p in offspring]

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if rnd.random() < cx_prob:
                child1, child2 = toolbox.recombination(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [toolbox.evaluate(ind) for ind in invalid_ind]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Generation of new population
        population[:] = offspring

        # Storing individuals to plot data
        best_ind.append(toolbox.clone(get_best_ind(population)[0]))
        all_ind.append([toolbox.clone(p) for p in population])
    return population, best_ind, all_ind


POP_SIZE = 1000
NB_GEN = 50
CX_PROB = 0.7


for i in range(1):
    population, best_ind, all_ind = gen_symb_expr(POP_SIZE, NB_GEN, CX_PROB)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    ax1.set_title("Fitness of the best individual for each iteration")
    ax1.set(xlabel="Iteration", ylabel="Fitness (sum of abs. errors)")
    ax1.plot([evaluate(ind) for ind in best_ind])

    ax2.set_title("Number of nodes of best individuals for each iteration")
    ax2.set(xlabel="Iteration", ylabel="Number of nodes")
    ax2.plot([len(ind) for ind in best_ind])

    ax3.set_title("Average sizes of individuals for each iteration")
    ax3.set(xlabel="Iteration", ylabel="Number of nodes")
    ax3.errorbar([i for i in range(NB_GEN+1)],
                 [average_node(pop) for pop in all_ind],
                 [[get_min_node(pop)[1] for pop in all_ind], [get_max_node(pop)[1] for pop in all_ind]])

    best_func = gp.compile(best_ind[-1], pset)
    ax4.set_title("Best function found vs Target function")
    ax4.plot(data.keys(), data.values(), 'o', label="Target")
    ax4.plot(data.keys(), [best_func(x) for x in data.keys()], label="Best ind: " + str(best_ind[-1]))
    fig.legend(loc="lower right")

    fig.show()
    print(best_ind[-1])
