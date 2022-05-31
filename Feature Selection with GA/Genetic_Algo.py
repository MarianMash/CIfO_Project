from copy import deepcopy
import random 
import pandas as pd
from charles.selection import fps, tournament,ranking
from charles.charles import Population, Individual
from charles.crossover import cycle_co, pmx_co, single_point_co,arithmetic_co
from charles.mutation import swap_mutation, inversion_mutation, binary_mutation
from data.fs_data import X,y

from sklearn.model_selection import train_test_split,cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier



def get_fitness(self):
    if self.representation.count(0) != len(self.representation):
        cols = [feature for index,feature in enumerate(X.columns) if self.representation[index] == 1]
        X_parsed = X[cols]
        X_train, X_test, y_train, y_test = train_test_split(X_parsed, y, test_size=0.20, random_state=42)

        l_reg = KNeighborsClassifier()

        l_reg.fit(X_train, y_train)

        prediction = l_reg.predict(X_test)
        fitness = accuracy_score(y_test,prediction)
        return fitness
    else:
        return 0

def the_shorter_the_better(self):
    ret = self.representation.count(1)
    return ret
       



Individual.get_fitness = get_fitness
Individual.second_contraint = the_shorter_the_better

selection = [fps,tournament,ranking]
cross_over = [single_point_co,pmx_co]
mutationas = [binary_mutation,swap_mutation, inversion_mutation]

pop = Population(
    size=31,
    sol_size=len(X.columns),
    valid_set=[0,1],
    replacement=True,
    optim="max",
)


one_frame = pop.evolve(
    gens=200,
    select=ranking,
    crossover=single_point_co,
    mutate=binary_mutation,
    co_p=0.9,
    mu_p=0.9,
    elitism=True,
            )