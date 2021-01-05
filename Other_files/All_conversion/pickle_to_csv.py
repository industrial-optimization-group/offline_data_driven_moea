from pygmo import fast_non_dominated_sorting as nds
import numpy as np
import pickle
import os
from joblib import Parallel, delayed
from non_domx import ndx
from optproblems import dtlz
from pymop.problems.welded_beam import WeldedBeam
from pymop.problems.truss2d import Truss2D
import matlab_wrapper2.matlab_wrapper as matlab_wrapper
import csv

dims = [10]
# dims = 4
############################################
folder_data = 'AM_Samples_109_Final'
#problem_testbench = 'DTLZ'
problem_testbench = 'DDMOPP'

#main_directory = 'Offline_Prob_DDMOPP3'
main_directory = 'Tests_Probabilistic_Finalx_new'
#main_directory = 'Tests_additional_obj1'
#main_directory = 'Tests_new_adapt'
#main_directory = 'Tests_toys'

#objectives = [2]
#objectives = [2,3,5]
objectives = [2,3,4,5,6,8,10]

#problems = ['DTLZ6']
#problems = ['DTLZ4','DTLZ5','DTLZ6','DTLZ7']
problems = ['P1','P2']
#problems = ['WELDED_BEAM'] #dims=4
#problems = ['TRUSS2D'] #dims=3

#modes = [1, 2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2, 7 = Approach_Prob
#modes = [7]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
modes = [1,7,8]

#sampling = ['BETA', 'MVNORM']
#sampling = ['LHS']
#sampling = ['BETA','OPTRAND','MVNORM']
#sampling = ['OPTRAND']
#sampling = ['MVNORM']
sampling = ['LHS', 'MVNORM']

#emo_algorithm = ['RVEA','IBEA']
emo_algorithm = ['RVEA']
#emo_algorithm = ['IBEA']
#emo_algorithm = ['NSGAIII']
#emo_algorithm = ['MODEL_CV']




#############################################

nruns = 11
pool_size = nruns


def f(name, num_of_objectives_real, num_of_variables, x):
    """The function to predict."""
    if name == "DTLZ1":
        obj_val = dtlz.DTLZ1(num_of_objectives_real, num_of_variables)(x)

    elif name == "DTLZ2":
        obj_val = dtlz.DTLZ2(num_of_objectives_real, num_of_variables)(x)

    elif name == "DTLZ3":
        obj_val = dtlz.DTLZ3(num_of_objectives_real, num_of_variables)(x)

    elif name == "DTLZ4":
        obj_val = dtlz.DTLZ4(num_of_objectives_real, num_of_variables)(x)

    elif name == "DTLZ5":
        obj_val = dtlz.DTLZ5(num_of_objectives_real, num_of_variables)(x)

    elif name == "DTLZ6":
        obj_val = dtlz.DTLZ6(num_of_objectives_real, num_of_variables)(x)

    elif name == "DTLZ7":
        obj_val = dtlz.DTLZ7(num_of_objectives_real, num_of_variables)(x)

    elif name == "WELDED_BEAM":
        problem_weld = WeldedBeam()
        F, G = problem_weld.evaluate(x)
        obj_val = F

    elif name == "TRUSS2D":
        problem_truss = Truss2D()
        F, G = problem_truss.evaluate(x)
        obj_val = F

    elif name == "DDMOPP_P1_2":
        matlab = matlab_wrapper.MatlabSession()
        matlab.put('x', x)
        matlab.eval('evaluate_DDMOPP')
        obj_val = matlab.get('y')

    return obj_val


def parallel_execute(run, path_to_file, prob, obj):
    actual_objectives_nds = None
    print(run)
    path_to_file = path_to_file + '/Run_' + str(run)
    infile = open(path_to_file, 'rb')
    results_data = pickle.load(infile)
    infile.close()
    individual_nds = results_data["individuals_solutions"]
    surrogate_objectives_nds = results_data["obj_solutions"]
    print(np.shape(individual_nds))
    path_to_file2 = path_to_file + '_pop'
    with open(path_to_file2, 'w') as f:
        writer = csv.writer(f)
        for line in individual_nds: writer.writerow(line)
    path_to_file3 = path_to_file + '_obj'
    with open(path_to_file3, 'w') as f:
        writer = csv.writer(f)
        for line in surrogate_objectives_nds: writer.writerow(line)
    print("File written...")



for samp in sampling:
    for obj in objectives:
        for n_vars in dims:
            for prob in problems:
                for algo in emo_algorithm:
                    for mode in modes:
                        path_to_file = main_directory \
                                       + '/Offline_Mode_' + str(mode) + '_' + algo + \
                                       '/' + samp + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                        print(path_to_file)

                        Parallel(n_jobs=pool_size)(
                            delayed(parallel_execute)(run, path_to_file, prob, obj) for run in range(nruns))
                        #for run in range(nruns):
                        #    parallel_execute(run, path_to_file, prob, obj)
