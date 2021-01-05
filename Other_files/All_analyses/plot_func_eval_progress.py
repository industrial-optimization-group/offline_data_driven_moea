import numpy as np
import pickle
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import csv
from IGD_calc import igd, igd_plus
from non_domx import ndx
from pygmo import hypervolume as hv
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
import math
from ranking_approaches import  calc_rank
from pymop.problems.welded_beam import WeldedBeam
from pymop.problems.truss2d import Truss2D
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from matplotlib import rc

#rc('font',**{'family':'serif','serif':['Helvetica']})
#rc('text', usetex=True)
#plt.rcParams.update({'font.size': 15})


pareto_front_directory = 'True_Pareto_5000'

mod_p_val = True
metric = 'HV'
#metric = 'IGD'
save_fig = 'pdf'
#dims = [5,8,10] #,8,10]
dims = [10]

folder_data = 'AM_Samples_109_Final'
#problem_testbench = 'DTLZ'
problem_testbench = 'DDMOPP'

#main_directory = 'Offline_Prob_DDMOPP3'
main_directory = 'Tests_Probabilistic_Finalx_new'
#main_directory = 'Tests_new_adapt'
#main_directory = 'Tests_toys'

objectives = [2]
#objectives = [3,5,6]
#objectives = [2,3,5]
#objectives = [2,3,4,5,6,8,10]

#problems = ['DTLZ2']
#problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
#problems = ['P1','P2']
problems = ['P1']
#problems = ['WELDED_BEAM'] #dims=4
#problems = ['TRUSS2D'] #dims=3

#modes = [1, 2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2, 7 = Approach_Prob
#modes = [0,7,70,71]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
#modes = [0,1,7,8]
#modes = [0,1,7,8]
modes = [1,9,7,8]
#modes = [1,2]
mode_length = int(np.size(modes))
#sampling = ['BETA', 'MVNORM']
sampling = ['LHS']
#sampling = ['BETA','OPTRAND','MVNORM']
#sampling = ['OPTRAND']
#sampling = ['MVNORM']
#sampling = ['LHS', 'MVNORM']

#emo_algorithm = ['RVEA','IBEA']
emo_algorithm = ['RVEA']
#emo_algorithm = ['IBEA']
#emo_algorithm = ['NSGAIII']
#emo_algorithm = ['MODEL_CV']




#approaches = ['Generic', 'Approach 1', 'Approach 2', 'Approach 3']
#approaches = ['Generic', 'Approach 1', 'Approach 2', 'Approach X']
#approaches = ['Initial sampling','Generic', 'Approach Prob','Approach Hybrid']
#approaches = ['Initial sampling','Generic_RVEA','Generic_IBEA']
#approaches = ['Initial sampling','Prob Old', 'Prob constant 1','Prob FE/FEmax']
#approaches = ['Generic', 'Approach Prob','Approach Hybrid']
#approaches = ['Generic', 'Probabilistic','Hybrid']
#approaches = ['7', '9', '11']
approaches = ['Initial sampling','Generic', 'TransferL','Probabilistic','Hybrid']
#"DTLZ6": {"2": [10, 10], "3": [10, 10, 10], "5": [10, 10, 10, 10, 10]},
#"DTLZ4": {"2": [4, 4], "3": [4, 4, 4], "5": [4, 4, 4, 4, 4]},
hv_ref = {"DTLZ2": {"2": [3, 3], "3": [3, 3, 3], "5": [3, 3, 3, 3, 3]},
          "DTLZ4": {"2": [3, 3.1], "3": [3, 3, 3], "5": [3, 3, 3, 3, 3]},
          "DTLZ5": {"2": [2.5, 3], "3": [2.5, 3, 3], "5": [2, 2, 2, 2, 2.5]},
          "DTLZ6": {"2": [10, 10], "3": [10, 10, 10], "5": [7, 7, 7, 7, 7]},
          "DTLZ7": {"2": [1, 20], "3": [1, 1, 30], "5": [1, 1, 1, 1, 45]}}


nruns = 11
f_eval_limit =1000
f_evals_per = 200
pool_size = nruns

plot_boxplot = True

l = [approaches]*nruns
labels = [list(i) for i in zip(*l)]
labels = [y for x in labels for y in x]

p_vals_all = None
index_all = None

def arg_median(a):
    if len(a) % 2 == 1:
        return np.where(a == np.median(a))[0][0]
    else:
        l,r = len(a) // 2 - 1, len(a) // 2
        left = np.partition(a, l)[l]
        right = np.partition(a, r)[r]
        return [np.where(a == left)[0][0], np.where(a == right)[0][0]]

for samp in sampling:
    for prob in problems:
        for obj in objectives:
            for n_vars in dims:
                fig = plt.figure(1, figsize=(10, 10))
                #fig = plt.figure()
                ax = fig.add_subplot(111)
                fig.set_size_inches(5, 5)

                #fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                #ax = fig.add_subplot(111)
                #fig.set_size_inches(5, 5)
                #plt.xlim(0, 1)
                #plt.ylim(0, 1)

                #if save_fig == 'pdf':
                #    plt.rcParams["text.usetex"] = True
                #with open(pareto_front_directory + '/True_5000_' + prob + '_' + obj + '.txt') as csv_file:
                #    csv_reader = csv.reader(csv_file, delimiter=',')
                if problem_testbench is 'DTLZ':
                    pareto_front = np.genfromtxt(pareto_front_directory + '/True_5000_' + prob + '_' + str(obj) + '.txt'
                                             , delimiter=',')


                #path_to_file = pareto_front_directory + '/' + 'Pareto_Weld'
                #infile = open(path_to_file, 'rb')
                #pareto_front = pickle.load(infile)
                #infile.close()
                #problem_weld = WeldedBeam()
                #pareto_front = problem_weld.pareto_front()

                for algo in emo_algorithm:
                    #igd_all = np.zeros([nruns, np.shape(modes)[0]])
                    igd_all = None
                    solution_ratio_all = None
                    for mode, mode_count in zip(modes,range(np.shape(modes)[0])):
                        if problem_testbench is 'DTLZ':
                            path_to_file = main_directory \
                                       + '/Offline_Mode_' + str(mode) + '_' + algo + \
                                       '/' + samp + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                        else:
                            path_to_file = main_directory \
                                           + '/Offline_Mode_' + str(mode) + '_' + algo + \
                                           '/' + samp + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                        print(path_to_file)

                        def igd_calc():
                            pass

                        def igd_box():
                            pass

                        def plot_median_run():
                            pass

                        def plot_convergence_plots():
                            pass


                        def parallel_execute(run, path_to_file):

                            if metric is 'IGD':
                                path_to_file = path_to_file + '/Run_' + str(run) + '_NDS'
                                infile = open(path_to_file, 'rb')
                                results_data=pickle.load(infile)
                                infile.close()
                                non_dom_front = results_data["actual_objectives_nds"]
                                non_dom_surr = results_data["surrogate_objectives_nds"]
                                #print(np.shape(non_dom_surr))
                                #print((np.max(non_dom_front,axis=0)))
                                solution_ratio = np.shape(non_dom_front)[0]/np.shape(non_dom_surr)[0]
                                return [igd(pareto_front, non_dom_front), solution_ratio]

                            else:
                                if problem_testbench is 'DDMOPP':
                                    path_to_file1 = path_to_file + '/Run_' + str(run+1) + '_soln_all'
                                    soln_all = []
                                    with open(path_to_file1,'r') as f:
                                        reader = csv.reader(f)
                                        for line in reader: soln_all.append(line)
                                    soln_all= np.array(soln_all, dtype=np.float32)
                                    ref = [2]*obj

                                else:
                                    path_to_file1 = path_to_file + '/Run_' + str(run+1) + '_soln'
                                    soln_all = []
                                    with open(path_to_file1,'r') as f:
                                        reader = csv.reader(f)
                                        for line in reader: soln_all.append(line)
                                    soln_all = np.array(soln_all, dtype=np.float32)
                                    ref = hv_ref[prob][str(obj)]

                                #print(actual_objectives_nds)
                                hv_iter = None
                                for f_iter in range(int(f_eval_limit/f_evals_per)):
                                    soln_all_temp = soln_all[f_iter*f_evals_per:(f_iter+1)*f_evals_per,:]
                                    if np.shape(soln_all_temp)[0] > 1:
                                        non_dom_front = ndx(soln_all_temp )
                                        soln_iter = soln_all_temp[non_dom_front[0][0]]
                                    else:
                                        soln_iter = soln_all_temp.reshape(1, obj)
                                    print(np.shape(hv_iter))
                                    solution_ratio = 0
                                    hyp = hv(soln_iter)
                                    hyp_temp = hyp.compute(ref)
                                    if f_iter == 0:
                                        hv_iter = hyp_temp
                                    else:
                                        hv_iter = np.vstack((hv_iter, hyp_temp))
                                print(hv_iter)
                                return hv_iter


                        #temp = Parallel(n_jobs=11)(delayed(parallel_execute)(run, path_to_file) for run in range(nruns))
                        #print(temp)
                        temp=None
                        for run in range(nruns):
                            temp=np.append(temp,parallel_execute(run, path_to_file))
                        print(temp)

                        #temp=np.asarray(temp)
                        #igd_temp = np.transpose(temp[:, 0])

"""
                        if plot_boxplot is True:
                            if igd_all is None:
                                igd_all = igd_temp
                                solution_ratio_all = solution_ratio_temp
                            else:
                                igd_all = np.vstack((igd_all, igd_temp))
                                solution_ratio_all = np.vstack((solution_ratio_all,solution_ratio_temp))
                    igd_all = np.transpose(igd_all)
                    solution_ratio_all = np.transpose(solution_ratio_all)
                    
                    p_value = np.zeros(int(math.factorial(mode_length)/((math.factorial(mode_length-2))*2)))
                    p_cor_temp = p_value
                    count = 0
                    count_prev = 0
                    for i in range(mode_length-1):
                        for j in range(i+1,mode_length):
                            if metric is 'HV':
                                #w, p = wilcoxon(x=igd_all[:, i], y=igd_all[:, j], alternative='less')
                                w, p = wilcoxon(x=igd_all[:, i], y=igd_all[:, j])
                            else:
                                w, p = wilcoxon(x=igd_all[:, i], y=igd_all[:, j], alternative='greater')
                            p_value[count] = p
                            count +=1
                        if mod_p_val is True:
                            r, p_cor_temp[count_prev:count], alps, alpb = multipletests(p_value[count_prev:count], alpha=0.05, method='bonferroni',
                                                                 is_sorted=False, returnsorted=False)
                            count_prev = count
                    p_cor = p_cor_temp
                    print(p_value)
                    print(p_cor)
                    if mod_p_val is False:
                        r, p_cor, alps, alpb = multipletests(p_value, alpha=0.05, method='bonferroni', is_sorted=False,
                                                             returnsorted=False)
                    current_index = [samp, n_vars, prob, obj]
                    #adding other indicators mean, median, std dev
                    #p_cor = (np.asarray([p_cor, np.mean(igd_all, axis=0),
                    #                             np.median(igd_all, axis=0),
                    #                             np.std(igd_all, axis=0)])).flatten()
                    ranking = calc_rank(p_cor, np.median(igd_all, axis=0),np.shape(modes)[0])
                    p_cor = np.hstack((p_cor, np.mean(igd_all, axis=0),
                                                 np.median(igd_all, axis=0),
                                                 np.std(igd_all, axis=0),ranking))
                    for medz in range(mode_length):
                        print("Median index:")
                        print(arg_median(igd_all[:, medz]))

                    p_cor = np.hstack((current_index,p_cor))
                    if p_vals_all is None:
                        p_vals_all = p_cor
                    else:
                        p_vals_all = np.vstack((p_vals_all, p_cor))

                    bp = ax.boxplot(igd_all, showfliers=False, widths=0.45)


                    #ax.set_title(metric + '_'+ samp + '_Algo_' + algo + '_' + prob + '_' + str(obj) + '_' + str(n_vars))
                    ax.set_title('Hypervolume comparison')
                    ax.set_xlabel('Approaches')
                    #ax.set_ylabel(metric)
                    ax.set_ylabel('Hypervolume')
                    ax.set_xticklabels(approaches, rotation=45, fontsize=15)
                    if problem_testbench is 'DTLZ':
                        filename_fig = main_directory + '/' + metric + '_'+ samp + '_' + algo + '_' + prob + '_' + str(obj)\
                                       + '_' + str(n_vars)
                    else:
                        filename_fig = main_directory + '/' + metric + '_' + samp + '_' + algo + '_' + prob + '_' + str(
                            obj) + '_' + str(n_vars)

                    if save_fig == 'png':
                        fig.savefig(filename_fig + '.png', bbox_inches='tight')
                    else:
                        fig.savefig(filename_fig + '.pdf', bbox_inches='tight')
                    ax.clear()

"""

#print(p_vals_all)
#if mod_p_val is False:
#    file_summary = main_directory + '/Summary_Quality_' + problem_testbench + '.csv'
#else:
#    file_summary = main_directory + '/Summary_Quality_ModP_' + problem_testbench + '.csv'
#with open(file_summary, 'w') as writeFile:
#    writer = csv.writer(writeFile)
#    writer.writerows(p_vals_all)
#writeFile.close()