from random import uniform, choice
from operator import attrgetter
import pandas as pd

from pandas import DataFrame
import numpy as np


def fps(population):
    """Fitness proportionate selection implementation.

    Args:
        population (Population): The population we want to select from.

    Returns:
        Individual: selected individual.
    """

    if population.optim == "max":
        # Sum total fitness
        total_fitness = sum([i.fitness for i in population])
        # Get a 'position' on the wheel
        spin = uniform(0, total_fitness)
        position = 0
        # Find individual in the position of the spin
        for individual in population:
            position += individual.fitness
            if position > spin:
                return individual

    elif population.optim == "min":
       

        #with 1- we can give small fitnesses bigger selection probability then bigger fitnesses
        fitness_sorted = {individual:1/individual.fitness for individual in population}

         # Sum total fitness
        total_fitness = sum(fitness_sorted.values())

        spin = uniform(0, total_fitness)
        position = 0
        # Find individual in the position of the spin
        for individual,val in fitness_sorted.items():
            position += val
            if position > spin:
                return individual

    else:
        raise Exception("No optimization specified (min or max).")
    
def ranking(population):
    """Ranking selection implementation.

    Args:
        population (Population): The population we want to select from.

    Returns:
        Individual: selected individual.
    """

    fitness_to_sort = {individual:individual.fitness for individual in population}

    pd_frame = pd.DataFrame.from_dict(fitness_to_sort, orient="index", columns = ["Fitness"]).reset_index()
    if population.optim == "max":
        pd_frame = pd_frame.sort_values("Fitness",ascending = True)
    elif population.optim == 'min':
       pd_frame = pd_frame.sort_values("Fitness",ascending = False)
        

    n = len(population)
    rank_sum = n * (n + 1) / 2

    selection_probabilities = [float(rank)/rank_sum for rank,some in enumerate(range(len(pd_frame)),1)]

    #based on the selection_probabilities we can select random an individuum
    prob_based_selec = np.random.choice(
    pd_frame["index"],1,
    p=selection_probabilities)

    return prob_based_selec[0].representation 

     



def tournament(population, size=10):
    """Tournament selection implementation.

    Args:
        population (Population): The population we want to select from.
        size (int): Size of the tournament.

    Returns:
        Individual: Best individual in the tournament.
    """

    # Select individuals based on tournament size
    tournament = [choice(population.individuals) for i in range(size)]
    # Check if the problem is max or min
    if population.optim == 'max':
        return max(tournament, key=attrgetter("fitness"))
    elif population.optim == 'min':
        return min(tournament, key=attrgetter("fitness"))
    else:
        raise Exception("No optimization specified (min or max).")

