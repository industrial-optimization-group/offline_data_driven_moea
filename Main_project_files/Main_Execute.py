from ..desdeo_problem.Problem import DataProblem
from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


from desdeo_problem.testproblems.TestProblems import test_problem_builder
from pyDOE import lhs
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main_project_files.SurrogateKriging import SurrogateKriging 
from desdeo_emo.EAs import MOEAD
from pygmo import non_dominated_front_2d as nd2
from non_domx import ndx
import scipy.io

from desdeo_problem.testproblems.TestProblems import test_problem_builder

max_samples = 50
max_iters = 50


def build_surrogates(problem_testbench, problem_name, nobjs, nvars, nsamples, is_data, x_data, y_data, surrogate_type, Z=None, z_samples=None):
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
    if surrogate_type == "generic_fullgp":
        problem.train(fgp)
    elif surrogate_type == "generic_sparsegp":
        problem.train(sgp,  model_parameters=z_samples)
    elif surrogate_type == "rf":
        problem.train(rf)
    elif surrogate_type == "htgp":
        problem.train(htgp)
    else:
        problem.train([sgp2]*nobjs, model_parameters=Z)

    end = time.time()
    time_taken = end - start
    return problem, time_taken

def read_dataset(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, run):
    mat = scipy.io.loadmat('./data/initial_samples/Initial_Population_' + problem_testbench + '_' + sampling +
                        '_AM_' + str(nvars) + '_'+str(nsamples)+'.mat')
    x = ((mat['Initial_Population_'+problem_testbench])[0][run])[0]
    if problem_testbench == 'DDMOPP':
        mat = scipy.io.loadmat('./data/initial_samples/Obj_vals_DDMOPP_'+sampling+'_AM_'+problem_name+'_'
                                + str(nobjs) + '_' + str(nvars)
                                + '_'+str(nsamples)+'.mat')
        y = ((mat['Obj_vals_DDMOPP'])[0][run])[0]
    elif problem_testbench == 'DTLZ':
        prob = test_problem_builder(
                    name=problem_name, n_of_objectives=nobjs, n_of_variables=nvars
                )
        y = prob.evaluate(x)[0]
    #elif problem_testbench == 'GAA':
    #    mat = scipy.io.loadmat('./'+folder_data+'/Obj_vals_GAA_'+sampling+'_AM_'+self.name+'_'
    #                            + str(self.num_of_objectives) + '_' + str(self.num_of_variables)
    #                            + '_'+str(sample_size)+'.mat')
    #    y = ((mat['Obj_vals_GAA'])[0][self.run])[0]
    return x, y

def optimize_surrogates(problem):
    print("Optimizing...")
    evolver_opt = RVEA(problem, use_surrogates=True, n_iterations=max_iters)
    while evolver_opt.continue_evolution():
        evolver_opt.iterate()
        print("Population size:",np.shape(evolver_opt.population.objectives)[0])
    #front_true = evolver_opt.population.objectives
    #evolver_opt.population.
    #print(front_true)
    return evolver_opt.population

def run_optimizer(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, is_data, surrogate_type, run):
    if surrogate_type == "strategy_1":
        results_dict = run_optimizer_strat1(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, is_data, surrogate_type, run)
    elif surrogate_type == "strategy_2":
        results_dict = run_optimizer_strat2(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, is_data, surrogate_type, run)
    elif surrogate_type == "strategy_3":
        results_dict = run_optimizer_strat3(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, is_data, surrogate_type, run)
    elif surrogate_type == "rf":
        results_dict = run_optimizer_rf(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, is_data, surrogate_type, run)
    elif surrogate_type == "htgp":
        results_dict = run_optimizer_htgp(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, is_data, surrogate_type, run)
    else:
        if is_data is True:
            x, y = read_dataset(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, run)
        surrogate_problem, time_taken = build_surrogates(problem_testbench,problem_name, nobjs, nvars, nsamples, is_data, x, y, surrogate_type,z_samples={'z_samples':max_samples})
        print(time_taken)
        population = optimize_surrogates(surrogate_problem)
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

def run_optimizer_strat1(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, is_data, surrogate_type, run):
    time_taken_all = 0
    inducing_inputs_dict = {}
    Z = []
    if is_data is True:
        x, y = read_dataset(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, run)

    # find non-domiated samples
    if np.shape(y)[0] > 1:
        non_dom_front = ndx(y)
        y_nd = y[non_dom_front[0][0]]
        x_nd = x[non_dom_front[0][0]]
    else:
        y_nd = y.reshape(1, nobjs)
        x_nd = x.reshape(1, nobjs)
    num_non_dom = np.shape(x_nd)[0]
    z_samples = max_samples - num_non_dom
    print("Non-dom samples = ")
    print(num_non_dom)
    print("1st model sample = ")
    print(z_samples)
    # build sparseGP model with 70% cost 
    surrogate_problem, time_taken = build_surrogates(problem_testbench,problem_name, nobjs, nvars, nsamples, is_data, x, y, "generic_sparsegp", z_samples={'z_samples':z_samples})
    # Add non-dominated samples strategy
    for i in range(nobjs):    
        inducing_inputs_dict['Z'] = np.vstack((np.asarray(surrogate_problem.objectives[i]._model.m.inducing_inputs),x_nd))
        Z.append(inducing_inputs_dict)
        inducing_inputs_dict={}

    #population = optimize_surrogates(surrogate_problem)    
    time_taken_all = time_taken_all + time_taken
    print(time_taken)
    # find K nearest neighbour samples with remaining 30% cost
    # ---- to be done later
  

    # choose few points by clustering ---- later

    # Build sparseGP with the combination of previous optimized points and KNN points
    surrogate_problem_2, time_taken = build_surrogates(problem_testbench,problem_name, nobjs, nvars, nsamples, is_data, x, y, "generic_sparsegp_strat1",Z=Z)
    time_taken_all = time_taken_all + time_taken
    print(time_taken)
    print("Total time taken:")
    print(time_taken_all)
    population = optimize_surrogates(surrogate_problem_2)
    
    results_dict = {
                'individual_archive': population.individuals_archive,
                'objectives_archive': population.objectives_archive,
                'uncertainty_archive': population.uncertainty_archive,
                'individuals_solutions': population.individuals,
                'obj_solutions': population.objectives,
                'uncertainty_solutions': population.uncertainity,
                'time_taken': time_taken_all
            }
    return results_dict

def run_optimizer_strat2(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, is_data, surrogate_type, run):
    time_taken_all = 0
    if is_data is True:
        x, y = read_dataset(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, run)
    mat = scipy.io.loadmat('./data/initial_samples/DDMOPP_HVPI_'+sampling+'_'+problem_name+'_'
                        + str(nobjs) + '_' + str(nvars)
                        + '_'+str(nsamples)+'.mat')
    hvpi = ((mat['hvpi'])[0][run])[0]
    start = time.time()
    Z = np.random.rand(max_samples,nvars)
    m_hvpi = GPy.models.SparseGPRegression(x,hvpi,Z=Z)
    m_hvpi.inducing_inputs.fix()
    m_hvpi.optimize('bfgs')
    m_hvpi.randomize()
    m_hvpi.Z.unconstrain()
    m_hvpi.optimize('bfgs')
    x_new = np.asarray(m_hvpi.inducing_inputs)
    #print("X new:")
    #print(x_new)
    z2 = []
    inducing_inputs_dict = {}
    for i in range(nobjs):    
        inducing_inputs_dict['Z'] = np.asarray(m_hvpi.inducing_inputs)
        z2.append(inducing_inputs_dict)
        inducing_inputs_dict={}
    #end = time.time()
    #time_taken = end - start
    #time_taken_all = time_taken_all + time_taken
    """
    y_new = None
    for i in range(max_samples):
        pos = np.where((x == x_new[i]).all(axis=1))
        y_ind=y[pos]
        print(y_ind)
        if y_new is None:
            y_new=y_ind
        else:
            y_new=np.vstack((y_new,y_ind))
    print("Y new:")
    print(y_new)
    
    surrogate_problem_final, time_taken = build_surrogates(problem_testbench,problem_name, nobjs, nvars, nsamples, is_data, x_new, y_new, "generic_fullgp",z_samples={'z_samples':max_samples})
    """
    surrogate_problem_final, time_taken = build_surrogates(problem_testbench,problem_name, nobjs, nvars, nsamples, is_data, x, y, "generic_sparsegp_strat1",Z=z2)   
    end = time.time()
    time_taken = end - start
    time_taken_all = time_taken_all + time_taken
    print("Total time taken:")
    print(time_taken_all)
    population = optimize_surrogates(surrogate_problem_final)
    
    results_dict = {
                'individual_archive': population.individuals_archive,
                'objectives_archive': population.objectives_archive,
                'uncertainty_archive': population.uncertainty_archive,
                'individuals_solutions': population.individuals,
                'obj_solutions': population.objectives,
                'uncertainty_solutions': population.uncertainity,
                'time_taken': time_taken_all
            }
    return results_dict

def run_optimizer_strat3(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, is_data, surrogate_type, run):
    start = time.time()
    time_taken_all = 0
    inducing_inputs_dict = {}
    inducing_inputs_dict_2={}
    prediction_all =  None
    prediction = []
    Z = []
    frac = 0.8
    print("Read dataset...")
    if is_data is True:
        x, y = read_dataset(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, run)
    print("Non dominated sorting...")
    # find non-domiated samples
    if np.shape(y)[0] > 1:
        non_dom_front_samples = ndx(y)
        y_nd = y[non_dom_front_samples[0][0]]
        x_nd = x[non_dom_front_samples[0][0]]
    else:
        y_nd = y.reshape(1, nobjs)
        x_nd = x.reshape(1, nobjs)
    num_non_dom_samples = np.shape(x_nd)[0]
    print("Non dominated samples")
    print(y_nd)
    z_samples = max_samples - num_non_dom_samples
    if z_samples <=max_samples*frac:
        z_samples = int(max_samples*frac)

    print("Non-dom samples = ")
    print(num_non_dom_samples)
    print("1st model samples = ")
    print(z_samples)
    print("Building first model...")
    # build sparseGP model with (max_samples - non_dom_samples) 
    surrogate_problem, time_taken = build_surrogates(problem_testbench,problem_name, nobjs, nvars, nsamples, is_data, x, y, "generic_sparsegp", z_samples={'z_samples': z_samples})
    print("Models built!")
    # get the samples used in the sparse model and their indices
    for i in range(nobjs):    
        inducing_inputs_dict[str(i)] = np.asarray(surrogate_problem.objectives[i]._model.m.inducing_inputs)
    print("The inducing inputs:")
    print(inducing_inputs_dict)
    # predict all the samples objective values
    for i in range(nobjs):
        for j in range(nsamples):
            pred = np.asarray(surrogate_problem.objectives[i]._model.predict(x[j].reshape(1,nvars))).reshape(1,2)[0,0]
            prediction.append(pred)
        prediction = np.asarray(prediction).reshape((2000,1))
        if prediction_all is None:
            prediction_all = prediction
        else:
            prediction_all = np.hstack((prediction_all, prediction))
        prediction = []
    y_pred = prediction_all
    print("Prediciton All:")
    print(y_pred)
    # perform non dom sort on the predicted samples
    if np.shape(y)[0] > 1:
        non_dom_front = ndx(y_pred)
        y_pred_nd = y_pred[non_dom_front[0][0]]
        x_pred_nd = x[non_dom_front[0][0]]
    else:
        y_pred_nd = y_pred.reshape(1, nobjs)
        x_pred_nd = x.reshape(1, nobjs)
        num_non_dom = np.shape(x_pred_nd)[0]
    print("Non dominated predictions")
    print(y_pred_nd)
    # check whether the actual non-dom samples are still non dominated. 
    non_dom_add_index = np.setxor1d(non_dom_front[0][0],non_dom_front_samples[0][0], assume_unique=True)    
    # if not, include them along with the sparse samples.
    if np.shape(non_dom_add_index)[0]>(max_samples-z_samples):
        non_dom_max = max_samples-z_samples
    else:
        non_dom_max = np.shape(non_dom_add_index)[0]
    x_add = x[non_dom_add_index[0:non_dom_max]]
    y_add = y[non_dom_add_index[0:non_dom_max]]
    for i in range(nobjs):    
        inducing_inputs_dict_2['Z'] = np.vstack((np.asarray(surrogate_problem.objectives[i]._model.m.inducing_inputs),x_add))
        Z.append(inducing_inputs_dict_2)
        print("Samples to be used for 2nd model:")
        print(np.shape(inducing_inputs_dict_2['Z']))
        inducing_inputs_dict_2={}
    # rebuild the models
    print("Building 2nd surogates...")   
    surrogate_problem_final, time_taken = build_surrogates(problem_testbench,problem_name, nobjs, nvars, nsamples, is_data, x, y, "generic_sparsegp_strat1",Z=Z) 
    print("Finished building 2nd surrogates!")
    # perform optimization
    end = time.time()
    time_taken = end - start
    time_taken_all = time_taken_all + time_taken
    print("Total time taken:")
    print(time_taken_all)
    population = optimize_surrogates(surrogate_problem_final)
    print("Optimization completed!")
    results_dict = {
                'individual_archive': population.individuals_archive,
                'objectives_archive': population.objectives_archive,
                'uncertainty_archive': population.uncertainty_archive,
                'individuals_solutions': population.individuals,
                'obj_solutions': population.objectives,
                'uncertainty_solutions': population.uncertainity,
                'time_taken': time_taken_all
            }
    return results_dict

def run_optimizer_rf(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, is_data, surrogate_type, run):
    time_taken_all = 0
    if is_data is True:
        x, y = read_dataset(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, run)
    surrogate_problem, time_taken = build_surrogates(problem_testbench,problem_name, nobjs, nvars, nsamples, is_data, x, y, "rf", z_samples={'z_samples':max_samples})
    print(time_taken)
    population = optimize_surrogates(surrogate_problem)
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

def run_optimizer_htgp(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, is_data, surrogate_type, run):
    time_taken_all = 0
    if is_data is True:
        x, y = read_dataset(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, run)

    start = time.time()
    print("Building trees...")
    surrogate_problem, time_taken = build_surrogates(problem_testbench,problem_name, nobjs, nvars, nsamples, is_data, x, y, "htgp")
    print("Building GPs...")
    total_points_all = 0
    total_points_all_sequence = []
    total_points_per_model = np.zeros(nobjs)
    delta_total_point = 1
    evolver_opt_tree = RVEA(surrogate_problem, use_surrogates=True, n_iterations=50, n_gen_per_iter=5)
    while evolver_opt_tree.continue_evolution() and delta_total_point > 0:
        evolver_opt_tree.iterate()    
        population_opt_tree = evolver_opt_tree.population
        X_solutions = population_opt_tree.individuals
        for i in range(nobjs):
            surrogate_problem.objectives[i]._model.addGPs(X_solutions)
            total_points_all += surrogate_problem.objectives[i]._model.total_point_gps
            total_points_per_model[i] = surrogate_problem.objectives[i]._model.total_point
        total_points_all_sequence = np.append(total_points_all_sequence,total_points_all)
        if evolver_opt_tree._iteration_counter > 5:
            delta_total_point = total_points_all - total_points_all_sequence[evolver_opt_tree._iteration_counter-3]
        print("Sequence:",total_points_all_sequence)
        print("Delta:",delta_total_point)
    end = time.time()
    time_taken = end - start
    print("Surrogates build complete in :",time_taken)
    print("Total points per model :", total_points_per_model)
    population = optimize_surrogates(surrogate_problem)

    ##### Previous HTGP
    #surrogate_problem, time_taken = build_surrogates(problem_testbench,problem_name, nobjs, nvars, nsamples, is_data, x, y, "htgp")
    #print(time_taken)
    #population = optimize_surrogates(surrogate_problem)
    #population = optimize_surrogates(surrogate_problem)
    results_dict = {
            'individual_archive': population.individuals_archive,
            'objectives_archive': population.objectives_archive,
            'uncertainty_archive': population.uncertainty_archive,
            'individuals_solutions': population.individuals,
            'obj_solutions': population.objectives,
            'uncertainty_solutions': population.uncertainity,
            'time_taken': time_taken,
            'total_points':  total_points_per_model
        }
    return results_dict