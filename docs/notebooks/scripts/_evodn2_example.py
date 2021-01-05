from pyrvea.Problem.dataproblem import DataProblem
from pyrvea.Problem.testproblem import TestProblem
from pyrvea.Population.Population import Population
from pyrvea.EAs.PPGA import PPGA
from pyrvea.EAs.RVEA import RVEA
from pyrvea.EAs.slowRVEA import slowRVEA
from pyrvea.EAs.NSGAIII import NSGAIII
import numpy as np
import pandas as pd
from copy import deepcopy
import plotly
import plotly.graph_objs as go

# test_prob = TestProblem(name="Fonseca-Fleming", num_of_variables=2)
# dataset, x, y = test_prob.create_training_data(samples=500)

dataset = pd.read_excel("ZDT1_1000.xls", header=0)
x = dataset.columns[0:30].tolist()
y = dataset.columns[30:].tolist()

problem = DataProblem(data=dataset, x=x, y=y)
problem.train_test_split(train_size=0.7)

# f_set = ("add", "sub", "mul", "div", "sqrt")
# t_set = [1, 2, 9, 29]
#
# ea_parameters = {
#     "generations_per_iteration": 20,
#     "iterations": 10,
#
# }
# model_params = {
#     "training_algorithm": RVEA,
#     "terminal_set": t_set,
#     "function_set": f_set,
#     "max_depth": 8
# }
#
# problem.train(
#     model_type="BioGP",
#     model_parameters=model_params,
#     ea_parameters=ea_parameters
#
# )

# ea_parameters = {
#     "generations_per_iteration": 15,
#     "iterations": 10,
#     "prob_mutation": 0.7,
#     "neighbourhood_radius": 5,
#     "target_pop_size": 100,
#     "mut_strength": 0.7
#
# }
#
# model_parameters = {
#     "training_algorithm": RVEA,
#
# }
#
# problem.train(
#     model_type="EvoDN2",
#     model_parameters=model_parameters,
#     ea_parameters=ea_parameters
#
# )

y_pred = problem.surrogates_predict(problem.data[problem.x])

problem.models["f1"][0].plot(y_pred[:, 0], problem.data["f1"], name="ZDT1_1000" + "f1")

problem.models["f2"][0].plot(y_pred[:, 1], problem.data["f2"], name="ZDT1_1000" + "f2")

# Optimize
# PPGA
pop_ppga = Population(problem)

ppga_parameters = {
    "prob_prey_move": 0.5,
    "prob_mutation": 0.5,
    "target_pop_size": 100,
    "kill_interval": 4,
    "iterations": 10,
    "predator_pop_size": 60,
    "generations_per_iteration": 10,
    "neighbourhood_radius": 5,
}

pop_ppga.evolve(EA=PPGA, ea_parameters=ppga_parameters)

pop_ppga.plot_pareto(
    name="Tests/" + problem.models["f1"][0].__class__.__name__ + "_ppga_" + "ZDT1_100"
)

# RVEA
pop_rvea = Population(
    problem,
    assign_type="LHSDesign",
    crossover_type="simulated_binary_crossover",
    mutation_type="bounded_polynomial_mutation",
    plotting=False,
)

rvea_parameters = {"iterations": 10, "generations_per_iteration": 25}

pop_rvea.evolve(EA=RVEA, ea_parameters=rvea_parameters)

pop_rvea.plot_pareto(
    name="Tests/" + problem.models["f1"][0].__class__.__name__ + "_rvea_" + "ZDT1_100"
)