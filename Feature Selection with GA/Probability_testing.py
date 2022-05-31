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

tournament_single_point_co_binary_mutation  = [tournament,single_point_co,binary_mutation]
ranking_single_point_co_binary_mutation = [ranking,single_point_co,binary_mutation]
tournament_pmx_co_binary_mutation = [tournament,pmx_co,binary_mutation]
combis = [tournament_single_point_co_binary_mutation,ranking_single_point_co_binary_mutation]
co_prob = [0.3,0.6]
mu_p = [0.3,0.6]

for comb in combis:
    for prob1 in co_prob:
        for mutatatioa in mu_p:


            sum_frame = pd.DataFrame()
            for i in range(20):
                pop = Population(
                    size=31,
                    sol_size=len(X.columns),
                    valid_set=[0,1],
                    replacement=True,
                    optim="max",
                )


                one_frame = pop.evolve(
                    gens=200,
                    select=comb[0],
                    crossover=comb[1],
                    mutate=comb[2],
                    co_p=prob1,
                    mu_p=mutatatioa,
                    elitism=True,
                            )

                sum_frame[f"Iteration_{i}"] = one_frame["Fitness"]
            
            sum_frame.to_csv(f"{prob1}_{mutatatioa}_{comb}.csv")