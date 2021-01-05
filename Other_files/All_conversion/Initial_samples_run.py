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
import scipy.io

dims = [10]
# dims = 4
############################################
folder_data = 'AM_Samples_109_Final'
main_directory = 'Tests_Probabilistic_Finalx'
# main_directory = 'O_Nautilus_Runs'
# main_directory = 'Tests_Final'
# main_directory = 'Tests_new_adapt'
# main_directory = 'Tests_toys'

# objectives = [3,5]
#objectives = [2, 3, 5]
#objectives = [2,3,4,5,6,8,10]
objectives = [10]

problem_testbench = 'DDMOPP'
#problem_testbench = 'DTLZ'

# problems = ['DTLZ2']
#problems = ['DTLZ2', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7']
problems = ['P1','P2']
# problems = ['DDMOPP_P1']
# problems = ['WELDED_BEAM'] #dims=4
# problems = ['TRUSS2D'] #dims=3

# modes = [1, 2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2, 7 = Approach_Prob
# modes = [2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
modes = [0]

# sampling = ['BETA', 'MVNORM']
# sampling = ['BETA']
# sampling = ['BETA','OPTRAND','MVNORM']
# sampling = ['OPTRAND']
# sampling = ['MVNORM']
sampling = ['LHS','MVNORM']

# emo_algorithm = ['RVEA','IBEA']
emo_algorithm = ['RVEA']
# emo_algorithm = ['IBEA']
# emo_algorithm = ['MODEL_CV']


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


def parallel_execute2(run, path_to_file, prob, obj, dims):
    path_to_file1 = path_to_file + '/Run_' + str(run) + '_soln'
    x = []
    with open(path_to_file1, 'r') as f:
        reader = csv.reader(f)
        for line in reader: x.append(line)


def parallel_execute(run, path_to_file, prob, obj, dims, mat, name, sampling):
    dataset = ((mat['Initial_Population_' + problem_testbench])[0][run])[0]
    if problem_testbench == 'DDMOPP':
        mat = scipy.io.loadmat('./' + folder_data + '/Obj_vals_DDMOPP_' + sampling + '_AM_' + name + '_'
                               + str(obj) + '_' + str(n_vars) + '_109.mat')
        y = ((mat['Obj_vals_DDMOPP'])[0][run])[0]
    else:
        y = None
        for ind in dataset:
            if y is None:
                y = np.asarray(f(name=prob, num_of_objectives_real=obj, num_of_variables=n_vars, x=ind))
            else:
                y = np.vstack((y, f(name=prob, num_of_objectives_real=obj, num_of_variables=n_vars, x=ind)))
    actual_objectives_nds = y
    print(run)
    individual_nds = dataset
    surrogate_objectives_nds = y
    if np.shape(individual_nds)[0] > 1:
        non_dom_front = ndx(actual_objectives_nds)
        actual_objectives_nds = actual_objectives_nds[non_dom_front[0][0]]
        actual_individual_nds = individual_nds[non_dom_front[0][0]]
    else:
        actual_objectives_nds = actual_objectives_nds.reshape(1, obj)
        actual_individual_nds = individual_nds.reshape(1, dims)
    print(np.amax(surrogate_objectives_nds, axis=0))
    print(np.amax(actual_objectives_nds, axis=0))
    results_dict = {
        'individual_nds': individual_nds,
        'surrogate_objectives_nds': surrogate_objectives_nds,
        'actual_individual_nds': actual_individual_nds,
        'actual_objectives_nds': actual_objectives_nds
    }
    # Write the non-dom sorted solutions
    if problem_testbench == 'DDMOPP':
        path_to_file = path_to_file + '/Run_' + str(run) + '_SOLN'
    else:
        path_to_file = path_to_file + '/Run_'+str(run)+'_NDS'
    outfile = open(path_to_file, 'wb')
    pickle.dump(results_dict, outfile)
    outfile.close()
    print("File written...")

    print(np.shape(actual_individual_nds))
    # print(results_dict)


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
                        if not os.path.exists(path_to_file):
                            os.makedirs(path_to_file)

                        mat = scipy.io.loadmat(
                            './' + folder_data + '/Initial_Population_' + problem_testbench + '_' + samp +
                            '_AM_' + str(n_vars) + '_109.mat')

                        """""
                        def parallel_execute_1(run, path_to_file):
                            print(run)
                            path_to_file = path_to_file + '/Run_' + str(run)
                            infile = open(path_to_file, 'rb')
                            #results_data = pickle.load(infile)
                            #infile.close()
                            print("Non-dominated sorting ...")
                            #xx = results_data["obj_solutions"]
                            #print(xx)

                            zz = ndx(np.array([[1,2],[2,4],[5,6]]))
                            print(zz)
                            #nxx = nds(np.array([[1,2],[2,4],[5,6]]))
                            #non_dom_front = nds(xx)

                            results_dict = {
                                'individual_nds': results_data["individuals_solutions"][non_dom_front[0][0]],
                                'objectives_nds': results_data["obj_solutions"][non_dom_front[0][0]]}
                            # Write the non-dom sorted solutions
                            path_to_file = path_to_file + '_NDS'
                            outfile = open(path_to_file, 'wb')
                            pickle.dump(results_dict, outfile)
                            outfile.close()
                            print("File written...")
                            print(np.size(non_dom_front[0][0]))
                            """
                        # Parallel(n_jobs=pool_size)(
                        #    delayed(parallel_execute)(run, path_to_file, prob, obj) for run in range(nruns))
                        for run in range(nruns):
                            parallel_execute(run, path_to_file, prob, obj, n_vars, mat, prob, samp)
