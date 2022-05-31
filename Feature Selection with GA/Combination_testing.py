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
selectio = [fps, tournament,ranking]
cross_over = [pmx_co,single_point_co]
mutationas = [binary_mutation,swap_mutation, inversion_mutation]
for selectionas in selectio:
    for cross_o in cross_over:
        for mutatatioa in mutationas:


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
                    select=selectionas,
                    crossover=cross_o,
                    mutate=mutatatioa,
                    co_p=0.9,
                    mu_p=0.9,
                    elitism=True,
                            )

                sum_frame[f"Iteration_{i}"] = one_frame["Fitness"]
            
            select1 = str(selectio)[10:].split(" ")[0]
            cross_o1 = str(cross_o)[10:].split(" ")[0]
            mutataa1 = str(mutatatioa)[10:].split(" ")[0]
            sum_frame.to_csv(f"{select1}_{cross_o1}_{mutataa1}.csv")