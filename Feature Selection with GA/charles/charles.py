from platform import mac_ver
from random import shuffle, choice, sample, random
from operator import attrgetter
from copy import deepcopy
from charles.utils import i_need_a_frame
import pandas as pd


class Individual:
    def __init__(
        self,
        representation=None,
        size=None,
        replacement=True,
        valid_set=None,
    ):
        if representation is None:
            if replacement == True:
                self.representation = [choice(valid_set) for i in range(size)]
            elif replacement == False:
                self.representation = sample(valid_set, size)
        else:
            pass
            if isinstance(representation,list):
                self.representation = representation
            else:
                self.representation = representation.representation
        self.fitness = self.get_fitness()
        self.second_contraint = self.second_contraint()

    def get_fitness(self):
        raise Exception("You need to monkey patch the fitness path.")

    def second_contraint(self):
        raise Exception("Something is wrong.")

    def index(self, value):
        return self.representation.index(value)

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, position):
        return self.representation[position]

    def __setitem__(self, position, value):
        self.representation[position] = value

    def __repr__(self):
        return f"Individual(size={len(self.representation)}); Fitness: {self.fitness}"


class Population:
    def __init__(self, size, optim, **kwargs):
        self.individuals = []
        self.size = size
        self.optim = optim
        for _ in range(size):
            self.individuals.append(
                Individual(
                    size=kwargs["sol_size"],
                    replacement=kwargs["replacement"],
                    valid_set=kwargs["valid_set"],
                )
            )

    def evolve(self, gens, select, crossover, mutate, co_p, mu_p, elitism):
        a_dict = dict()
        for gen in range(gens):
            new_pop = []

            if elitism == True:
                if self.optim == "max":
                    elite = deepcopy(max(self.individuals, key=attrgetter("fitness")))
                    elite_fitness = elite.fitness
                    all_max = [indiv for indiv in self.individuals if indiv.fitness == elite_fitness]
                    elite = min(all_max, key=attrgetter("second_contraint"))
                elif self.optim == "min":
                    elite = deepcopy(min(self.individuals, key=attrgetter("fitness")))
                    elite_fitness = elite.fitness
                    all_max = [indiv for indiv in self.individuals if indiv.fitness == elite_fitness]
                    elite = max(all_max, key=attrgetter("second_contraint"))

            while len(new_pop) < self.size:
                parent1, parent2 = select(self), select(self)
                # Crossover
                if random() < co_p:
                    offspring1, offspring2 = crossover(parent1, parent2)

                    
                else:
                    offspring1, offspring2 = parent1, parent2
                # Mutation
                if random() < mu_p:
                    offspring1 = mutate(offspring1)
                if random() < mu_p:
                    offspring2 = mutate(offspring2)

                new_pop.append(Individual(representation=offspring1))
                if len(new_pop) < self.size:
                    new_pop.append(Individual(representation=offspring2))

            if elitism == True:
                if self.optim == "max":
                    least = min(new_pop, key=attrgetter("fitness"))
                    least_fitness = least.fitness
                    all_max = [indiv for indiv in new_pop if indiv.fitness == least_fitness]
                    least = max(all_max, key=attrgetter("second_contraint"))


                elif self.optim == "min":
                    least = max(new_pop, key=attrgetter("fitness"))
                    least_fitness = least.fitness
                    all_max = [indiv for indiv in new_pop if indiv.fitness == least_fitness]
                    least = min(all_max, key=attrgetter("second_contraint"))

                new_pop.pop(new_pop.index(least))
                new_pop.append(elite)

            self.individuals = new_pop
            if self.optim == "max":
                best_one = max(self, key=attrgetter("fitness"))
                print(f'Best Individual: {best_one.representation} with Fitness: {best_one.fitness}')
                a_dict[gen] = best_one.fitness

                
            elif self.optim == "min":
                best_one = min(self, key=attrgetter("fitness"))
                print(f'Best Individual: {best_one.representation} with Fitness: {best_one.fitness}')
                a_dict[gen] = best_one.fitness

            if gen == gens-1:
                pd_frame = pd.DataFrame.from_dict(a_dict, orient="index", columns = ["Fitness"]).reset_index()
                return pd_frame


                

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]

    def __repr__(self):
        return f"Population(size={len(self.individuals)}, individual_size={len(self.individuals[0])})"

