from desdeo_problem.Problem import DataProblem
from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from desdeo_problem.surrogatemodels.SurrogateKriging import SurrogateKriging


from desdeo_problem.testproblems.TestProblems import test_problem_builder
from pyDOE import lhs
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from desdeo_emo.EAs.ProbRVEA import RVEA

from desdeo_emo.EAs.ProbRVEA import ProbRVEA
from desdeo_emo.EAs.ProbRVEA import ProbRVEA_v3

from desdeo_emo.EAs.ProbRVEA import HybRVEA
from desdeo_emo.EAs.ProbRVEA import HybRVEA_v3

from desdeo_emo.EAs.ProbMOEAD import MOEA_D

from desdeo_emo.EAs.ProbMOEAD import ProbMOEAD
from desdeo_emo.EAs.ProbMOEAD import ProbMOEAD_v3

from desdeo_emo.EAs.ProbMOEAD import HybMOEAD
from desdeo_emo.EAs.ProbMOEAD import HybMOEAD_v3
#from pygmo import non_dominated_front_2d as nd2
#from non_domx import ndx
import scipy.io
from sklearn.neighbors import NearestNeighbors
import time

gen_per_iter_set = 10
max_func_evals = 40000
nsamples = 109

def build_surrogates(problem_testbench, problem_name, nobjs, nvars, is_data, x_data, y_data):
    x_names = [f'x{i}' for i in range(1,nvars+1)]
    y_names = [f'f{i}' for i in range(1,nobjs+1)]
    row_names = ['lower_bound','upper_bound']
    if is_data is False:
        prob = test_problem_builder(problem_name, nvars, nobjs)
        x = lhs(nvars, nsamples)
        y = prob.evaluate(x)
        data = pd.DataFrame(np.hstack((x,y.objectives)), columns=x_names+y_names)
    else:
        data = pd.DataFrame(np.hstack((x_data,y_data)), columns=x_names+y_names)
    if problem_testbench == 'DDMOPP':
        x_low = np.ones(nvars)*-1
        x_high = np.ones(nvars)
    elif problem_testbench == 'DTLZ':
        x_low = np.ones(nvars)*0
        x_high = np.ones(nvars)    
    bounds = pd.DataFrame(np.vstack((x_low,x_high)), columns=x_names, index=row_names)
    problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names,bounds=bounds)
    start = time.time()
    problem.train(SurrogateKriging)
    end = time.time()
    time_taken = end - start
    return problem, time_taken

def read_dataset(problem_testbench, folder_data, problem_name, nobjs, nvars, sampling, run):
    if problem_testbench == "DDMOPP":
        mat = scipy.io.loadmat(folder_data + '/Initial_Population_' + problem_testbench + '_' + sampling +
                            '_AM_' + str(nvars) + '_109.mat')
        x = ((mat['Initial_Population_'+problem_testbench])[0][run])[0]
        mat = scipy.io.loadmat(folder_data+'/Obj_vals_DDMOPP_'+sampling+'_AM_'+problem_name+'_'
                                       + str(nobjs) + '_' + str(nvars) + '_109.mat')
        y = ((mat['Obj_vals_DDMOPP'])[0][run])[0]
    else:
        mat = scipy.io.loadmat(folder_data + '/Initial_Population_DTLZ_'+sampling+'_AM_' + str(nvars) + '_109.mat')
        prob = test_problem_builder(
            name=problem_name, n_of_objectives=nobjs, n_of_variables=nvars
        )
        x = ((mat['Initial_Population_DTLZ'])[0][run])[0]
        y = prob.evaluate(x)[0]
    return x, y

def optimize_surrogates_1(problem,x):
    print("Optimizing...")
    evolver_opt = RVEA(problem, use_surrogates=True, n_gen_per_iter=gen_per_iter_set, total_function_evaluations=max_func_evals) #, population_params={'design':'InitSamples','init_pop':x}, population_size=109)
    while evolver_opt.continue_evolution():
        evolver_opt.iterate()
        print("FE count:",evolver_opt._function_evaluation_count)
    #front_true = evolver_opt.population.objectives
    #evolver_opt.population.
    #print(front_true)
    return evolver_opt.population

def optimize_surrogates_7(problem,x):
    print("Optimizing...")
    evolver_opt = ProbRVEA_v3(problem, use_surrogates=True, n_gen_per_iter=gen_per_iter_set, total_function_evaluations=max_func_evals) #, population_params={'design':'InitSamples','init_pop':x}, population_size=109)
    while evolver_opt.continue_evolution():
        evolver_opt.iterate()
        print("FE count:",evolver_opt._function_evaluation_count)
    #front_true = evolver_opt.population.objectives
    #evolver_opt.population.
    #print(front_true)
    return evolver_opt.population

def optimize_surrogates_8(problem,x):
    print("Optimizing...")
    evolver_opt = HybRVEA_v3(problem, use_surrogates=True, n_gen_per_iter=gen_per_iter_set, total_function_evaluations=max_func_evals) #, population_params={'design':'InitSamples','init_pop':x}, population_size=109)
    while evolver_opt.continue_evolution():
        evolver_opt.iterate()
        print("FE count:",evolver_opt._function_evaluation_count)
    #front_true = evolver_opt.population.objectives
    #evolver_opt.population.
    #print(front_true)
    return evolver_opt.population

def optimize_surrogates_12(problem,x):
    print("Optimizing...")
    evolver_opt = MOEA_D(problem, use_surrogates=True, n_gen_per_iter=gen_per_iter_set, total_function_evaluations=max_func_evals) #, population_params={'design':'InitSamples','init_pop':x}, population_size=109)
    while evolver_opt.continue_evolution():
        evolver_opt.iterate()
        print("FE count:",evolver_opt._function_evaluation_count)
    #front_true = evolver_opt.population.objectives
    #evolver_opt.population.
    #print(front_true)
    return evolver_opt.population

def optimize_surrogates_72(problem,x):
    print("Optimizing...")
    evolver_opt = ProbMOEAD_v3(problem, use_surrogates=True, n_gen_per_iter=gen_per_iter_set, total_function_evaluations=max_func_evals) #, population_params={'design':'InitSamples','init_pop':x}) #, population_size=109)
    while evolver_opt.continue_evolution():
        evolver_opt.iterate()
        print("FE count:",evolver_opt._function_evaluation_count)
    #front_true = evolver_opt.population.objectives
    #evolver_opt.population.
    #print(front_true)
    return evolver_opt.population

def optimize_surrogates_82(problem,x):
    print("Optimizing...")
    evolver_opt = HybMOEAD_v3(problem, use_surrogates=True, n_gen_per_iter=gen_per_iter_set, total_function_evaluations=max_func_evals) #, population_params={'design':'InitSamples','init_pop':x}, population_size=109)
    while evolver_opt.continue_evolution():
        evolver_opt.iterate()
        print("FE count:",evolver_opt._function_evaluation_count)
    #front_true = evolver_opt.population.objectives
    #evolver_opt.population.
    #print(front_true)
    return evolver_opt.population

def run_optimizer(problem_testbench, folder_data, problem_name, nobjs, nvars, sampling, is_data, run, approach):
    if is_data is True:
        x, y = read_dataset(problem_testbench, folder_data, problem_name, nobjs, nvars, sampling, run)
    surrogate_problem, time_taken = build_surrogates(problem_testbench,problem_name, nobjs, nvars, is_data, x, y)
    print(time_taken)
    if approach == 1:
        population = optimize_surrogates_1(surrogate_problem,x)
    elif approach == 7:
        population = optimize_surrogates_7(surrogate_problem,x)
    elif approach == 8:
        population = optimize_surrogates_8(surrogate_problem,x)
    elif approach == 12:
        population = optimize_surrogates_12(surrogate_problem,x)
    elif approach == 72:
        population = optimize_surrogates_72(surrogate_problem,x)
    elif approach == 82:
        population = optimize_surrogates_82(surrogate_problem,x)
    results_dict = {
            'individual_archive': population.individuals_archive,
            'objectives_archive': population.objectives_archive,
            'uncertainty_archive': population.uncertainty_archive,
            'individuals_solutions': population.individuals,
            'obj_solutions': population.objectives,
            'uncertainty_solutions': population.uncertainity,
            'time_taken': time_taken
        }
    return results_dict
