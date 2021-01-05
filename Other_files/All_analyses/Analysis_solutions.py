import numpy as np
import pickle
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import csv
from IGD_calc import igd, igd_plus
from pymop.problems.welded_beam import WeldedBeam
from pymop.problems.truss2d import Truss2D
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from optproblems import dtlz
from non_domx import ndx
import pandas as pd
from scipy.spatial import distance
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
from matplotlib import rc
from ranking_approaches import  calc_rank
import math

rc('font',**{'family':'serif','serif':['Helvetica']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 15})

pareto_front_directory = 'True_Pareto_5000'


#dims = [5,8,10]
dims = [10]
mod_p_val = True

main_directory = 'Tests_Probabilistic_Finalx_new'
#main_directory = 'Tests_additional_obj1'
#main_directory = 'O_Nautilus_Runs'
#main_directory = 'Tests_Final'
#main_directory = 'Tests_new_adapt'
#main_directory = 'Tests_toys'

#objectives = [3,5]
objectives = [2,3,5]
#objectives = [6]
#objectives = [2,3,4,5,6,8,10]

#problems = ['DTLZ2']
problems = ['DTLZ2', 'DTLZ4','DTLZ5','DTLZ6','DTLZ7']
#problems = ['DDMOPP_P1']
#problems = ['P1','P2']
#problems = ['P2']
#problems = ['WELDED_BEAM'] #dims=4
#problems = ['TRUSS2D'] #dims=3

#modes = [1, 2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2, 7 = Approach_Prob
#modes = [7]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
#modes = [1,2]
modes = [1,9,7,8]

#sampling = ['BETA', 'MVNORM']
#sampling = ['BETA']
#sampling = ['BETA','OPTRAND','MVNORM']
#sampling = ['OPTRAND']
#sampling = ['MVNORM']
sampling = ['LHS','MVNORM']
#sampling = ['LHS']

#emo_algorithm = ['RVEA','IBEA']
emo_algorithm = ['RVEA']
#emo_algorithm = ['IBEA']
#emo_algorithm = ['MODEL_CV']



#approaches = ['Generic', 'Probabilistic', 'Hybrid']
#approaches = ['Generic', 'Approach 1', 'Approach 2', 'Approach X']
#approaches = ['Generic', 'Approach 1']
#approaches = ['Generic', 'TransferL','Probabilistic','Hybrid']
approaches = ['Gen','TL','Prob','Hyb']

problem_testbench = 'DTLZ'
#problem_testbench = 'DDMOPP'

nruns = 11
pool_size = nruns
rmsemvr = True
plot_boxplot = True
p_vals_all = None

def objf(name, num_of_objectives_real,  num_of_variables, x):
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

    return obj_val

for samp in sampling:
    for n_vars in dims:
        for prob in problems:
            for algo in emo_algorithm:
                for obj in objectives:
                    fig = plt.figure(1, figsize=(10, 10))
                    ax = fig.add_subplot(111)
                    fig.set_size_inches(4, 4)
                    mean_r_sq_cv_all = None
                    mean_rmse_cv_all = None
                    #igd_all = np.zeros([nruns, np.shape(modes)[0]])
                    for mode, mode_count in zip(modes,range(np.shape(modes)[0])):
                        path_to_file = main_directory \
                                   + '/Offline_Mode_' + str(mode) + '_' + algo + \
                                   '/' + samp + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                        print(path_to_file)


                        def parallel_execute(run, path_to_file):
                            actual_objectives_nds = None
                            rmse_sols = None
                            rmse_mv_sols = 0
                            if mode == 9:
                                path_to_file2 = path_to_file + '/Run_' + str(run + 1) + '_soln'
                                actual_objectives_nds = []
                                with open(path_to_file2, 'r') as f:
                                    reader = csv.reader(f)
                                    for line in reader: actual_objectives_nds.append(line)
                                path_to_file2 = path_to_file + '/Run_' + str(run + 1) + '_surr'
                                surrogate_objectives_nds = []
                                with open(path_to_file2, 'r') as f:
                                    reader = csv.reader(f)
                                    for line in reader: surrogate_objectives_nds.append(line)
                                actual_objectives_nds = np.asarray(actual_objectives_nds, dtype=np.float32)
                                surrogate_objectives_nds = np.asarray(surrogate_objectives_nds, dtype=np.float32)
                                print(np.shape(actual_objectives_nds))
                                print(np.shape(surrogate_objectives_nds))

                            else:
                                if problem_testbench == 'DTLZ':
                                    path_to_file = path_to_file + '/Run_' + str(run) + '_NDS'
                                    infile = open(path_to_file, 'rb')
                                    results_data=pickle.load(infile)
                                    infile.close()
                                    individual_nds = np.asarray(results_data['individual_nds'])
                                    surrogate_objectives_nds = np.asarray(results_data['surrogate_objectives_nds'])

                                    #fig = plt.figure(1, figsize=(6, 6))
                                    #ax = fig.add_subplot(111)
                                    #plt.scatter(surrogate_objectives_nds[:,0],surrogate_objectives_nds[:,1])
                                    #plt.show()

                                    for i in range(np.shape(individual_nds)[0]):
                                        if i == 0:
                                            actual_objectives_nds = np.asarray(objf(prob, obj, n_vars, individual_nds[i, :]))
                                        else:
                                            actual_objectives_nds = np.vstack(
                                                (actual_objectives_nds, objf(prob, obj, n_vars, individual_nds[i, :])))
                                else:
                                    path_to_file1 = path_to_file + '/Run_' + str(run) + '_obj'
                                    surr_obj = []
                                    with open(path_to_file1, 'r') as f:
                                        reader = csv.reader(f)
                                        for line in reader: surr_obj.append(line)
                                    surr_obj = list(surr_obj)
                                    surr_obj = [[float(y) for y in x] for x in surr_obj]
                                    surr_obj = np.array(surr_obj)

                                    path_to_file1 = path_to_file + '/Run_' + str(run) + '_soln'
                                    actual_obj = []
                                    with open(path_to_file1, 'r') as f:
                                        reader = csv.reader(f)
                                        for line in reader: actual_obj.append(line)
                                    actual_obj = list(actual_obj)
                                    actual_obj = [[float(y) for y in x] for x in actual_obj]
                                    actual_obj = np.array(actual_obj)
                                    print(run)
                                    print(np.shape(actual_obj))
                                    print(np.shape(surr_obj))
                                    #non_dom_front = ndx(actual_obj)
                                    #actual_objectives_nds = actual_obj[non_dom_front[0][0]]
                                    #surrogate_objectives_nds = surr_obj[non_dom_front[0][0]]
                                    actual_objectives_nds = actual_obj
                                    surrogate_objectives_nds = surr_obj

                            if rmsemvr is False:
                                for i in range(obj):
                                    if i==0:
                                        rmse_sols=np.sqrt(mean_squared_error(surrogate_objectives_nds[:,i],actual_objectives_nds[:,i]))
                                    else:
                                        rmse_sols=np.vstack((rmse_sols,
                                                            np.sqrt(mean_squared_error(surrogate_objectives_nds[:,i],actual_objectives_nds[:,i]))))
                            else:
                                # calculating multivariate RMSE
                                for i in range(np.shape(actual_objectives_nds)[0]):
                                    rmse_mv_sols += distance.euclidean(surrogate_objectives_nds[i,:obj],actual_objectives_nds[i,:])
                                rmse_mv_sols = rmse_mv_sols/np.shape(actual_objectives_nds)[0]
                            if rmsemvr is True:
                                return rmse_mv_sols
                            else:
                                return rmse_sols


                        rmse_temp = Parallel(n_jobs=11)(delayed(parallel_execute)(run, path_to_file) for run in range(nruns))
                        rmse_temp = np.asarray(rmse_temp)
                        #mean_r_sq_cv_temp = temp[:][:, 0]
                        #mean_rmse_cv_temp = temp[:][:, 0]
                        if rmsemvr is True:
                            mean_rmse_cv_temp = np.reshape(rmse_temp,(nruns,1))
                        else:
                            mean_rmse_cv_temp = np.reshape(rmse_temp,(nruns,obj))
                        #print(mean_r_sq_cv_temp)
                        #for run in range(nruns):
                        #    parallel_execute(run, path_to_file)

                        if plot_boxplot is True:
                            if mean_rmse_cv_all is None:
                                #mean_r_sq_cv_all = mean_r_sq_cv_temp
                                mean_rmse_cv_all = mean_rmse_cv_temp
                            else:
                                #mean_r_sq_cv_all = np.hstack((mean_r_sq_cv_all, mean_r_sq_cv_temp))
                                mean_rmse_cv_all = np.hstack((mean_rmse_cv_all, mean_rmse_cv_temp))

                    print(mean_rmse_cv_all)
                    count = 0
                    mode_size = len(modes)
                    mode_length = mode_size
                    obj_samp = []
                    if rmsemvr is False:
                        mean_rmse_cv_all_temp = np.zeros((11, obj*mode_size))
                        for i in range(obj):
                            for j in range(mode_size):
                                mean_rmse_cv_all_temp[:, count] = mean_rmse_cv_all[:, i+j*obj]
                                count = count+1
                                obj_samp.append(approaches[j]+'_Obj'+str(i))
                        mean_rmse_cv_all = mean_rmse_cv_all_temp
                    else:
                        obj_samp = approaches

                    #bp = ax.boxplot(mean_rmse_cv_all, showfliers=False)
                    bp = ax.boxplot(mean_rmse_cv_all, showfliers=False, widths=0.45)
                    if rmsemvr is False:
                        ax.set_title('RMSE_Solutions_' + prob + '_' + str(obj))
                        ax.set_xlabel('Objectives')
                        ax.set_ylabel('RMSE')
                        ax.set_xticklabels(obj_samp, rotation=45, fontsize=8)
                        filename_fig = main_directory + '/RMSE_' + samp + '_' + algo + '_' + prob + '_' + str(obj) + '_' + str(n_vars) + '.png'
                        fig.savefig(filename_fig, bbox_inches='tight')
                        ax.clear()
                    else:
                        ax.set_title('RMSE_MVR_Solutions_' + prob + '_' + str(obj))
                        ax.set_title('RMSE comparison')
                        ax.set_xlabel('Approaches')
                        ax.set_ylabel('RMSE')
                        ax.set_xticklabels(obj_samp, rotation=45, fontsize=15)
                        filename_fig = main_directory + '/RMSE_MVR_' + samp + '_' + algo + '_' + prob + '_' + str(obj) + '_' + str(n_vars) + '.pdf'
                        fig.savefig(filename_fig, bbox_inches='tight')
                        ax.clear()

                    p_value =  np.zeros(int(math.factorial(mode_length)/((math.factorial(mode_length-2))*2)))
                    print(p_value)
                    p_cor_temp = p_value
                    count = 0
                    count_prev = 0
                    for i in range(np.size(modes) - 1):
                        for j in range(i + 1, np.size(modes)):
                            #w, p = wilcoxon(x=mean_rmse_cv_all[:, i], y=mean_rmse_cv_all[:, j], alternative='greater')
                            w, p = wilcoxon(x=mean_rmse_cv_all[:, i], y=mean_rmse_cv_all[:, j])
                            p_value[count] = p
                            count += 1
                        if mod_p_val is True:
                            r, p_cor_temp[count_prev:count], alps, alpb = multipletests(p_value[count_prev:count],
                                                                                        alpha=0.05,
                                                                                        method='bonferroni',
                                                                                        is_sorted=False,
                                                                                        returnsorted=False)
                            count_prev = count
                    p_cor = p_cor_temp
                    if mod_p_val is False:
                        r, p_cor, alps, alpb = multipletests(p_value, alpha=0.05, method='bonferroni', is_sorted=False,
                                                         returnsorted=False)
                    current_index = [samp, n_vars, prob, obj]
                    ranking = calc_rank(p_cor, np.median(mean_rmse_cv_all, axis=0)*-1, np.shape(modes)[0])
                    p_cor = np.hstack((p_cor, np.mean(mean_rmse_cv_all, axis=0),
                                                 np.median(mean_rmse_cv_all, axis=0),
                                                 np.std(mean_rmse_cv_all, axis=0),ranking))
                    p_cor = np.hstack((current_index, p_cor))
                    if p_vals_all is None:
                        p_vals_all = p_cor
                    else:
                        p_vals_all = np.vstack((p_vals_all, p_cor))
                    """""
                    igd_all = np.transpose(igd_all)
                    solution_ratio_all = np.transpose(solution_ratio_all)
                    #print(np.max(igd_all))
                    #print(np.max(solution_ratio_all))
                    #print(igd_all)
                    #print(solution_ratio_all)
                    bp = ax.boxplot(igd_all, showfliers=False)
                    ax.set_title('IGD_'+ samp + '_Algo_' + algo + '_' + prob + '_' + str(obj))
                    ax.set_xlabel('Approaches')
                    ax.set_ylabel('IGD')
                    ax.set_xticklabels(approaches, rotation=45, fontsize=8)
                    filename_fig = main_directory + '/IGD_'+ samp + '_' + algo + '_' + prob + '_' + str(obj) + '.png'
                    fig.savefig(filename_fig, bbox_inches='tight')
                    ax.clear()
                    #bp = ax.boxplot(solution_ratio_all, showfliers=False)
                    #filename_fig = main_directory + '/SolnRatio_' + samp + '_' + algo + '_' + prob + '_' + str(obj) + '.png'
                    #fig.savefig(filename_fig, bbox_inches='tight')
                    #ax.clear()
                    """
print(p_vals_all)
if mod_p_val is False:
    file_summary = main_directory + '/Summary_RMSE_' + problem_testbench + '.csv'
else:
    file_summary = main_directory + '/Summary_RMSE_ModP_' + problem_testbench + '.csv'
with open(file_summary, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(p_vals_all)
writeFile.close()