import numpy as np
import pickle
import os
from joblib import Parallel, delayed
import seaborn as sns;

sns.set()
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
import pandas as pd
import csv
from IGD_calc import igd, igd_plus
from non_domx import ndx
from pygmo import hypervolume as hv
from scipy.spatial import distance
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
import math
from ranking_approaches import calc_rank
from pymop.problems.welded_beam import WeldedBeam
from pymop.problems.truss2d import Truss2D
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from matplotlib import rc

# rc('font',**{'family':'serif','serif':['Helvetica']})
# rc('text', usetex=True)
# plt.rcParams.update({'font.size': 15})


pareto_front_directory = 'True_Pareto_5000'

mod_p_val = True

save_fig = 'pdf'
# dims = [5,8,10] #,8,10]
dims = [10]

folder_data = 'AM_Samples_109_Final'
problem_testbench = 'DTLZ'
#problem_testbench = 'DDMOPP'

# main_directory = 'Offline_Prob_DDMOPP3'
main_directory = 'Tests_Probabilistic_Finalx_new'
# main_directory = 'Tests_new_adapt'
# main_directory = 'Tests_toys'

#objectives = [5]
# objectives = [3,5,6]
objectives = [2,3,5]
#objectives = [2, 3, 4, 5, 6, 8, 10]

# problems = ['DTLZ2']
problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
#problems = ['P1', 'P2']
#problems = ['P1']
# problems = ['WELDED_BEAM'] #dims=4
# problems = ['TRUSS2D'] #dims=3

# modes = [1, 2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2, 7 = Approach_Prob
# modes = [0,7,70,71]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
# modes = [0,1,7,8]
#modes = [7,8]
modes = [1, 9, 7, 8]
# modes = [1,2]
mode_length = int(np.size(modes))
# sampling = ['BETA', 'MVNORM']
#sampling = ['LHS']
# sampling = ['BETA','OPTRAND','MVNORM']
# sampling = ['OPTRAND']
#sampling = ['MVNORM']
sampling = ['LHS', 'MVNORM']
sx='MVNS'
#sx='LHS'
# emo_algorithm = ['RVEA','IBEA']
emo_algorithm = ['RVEA']
# emo_algorithm = ['IBEA']
# emo_algorithm = ['NSGAIII']
# emo_algorithm = ['MODEL_CV']


# approaches = ['Generic', 'Approach 1', 'Approach 2', 'Approach 3']
# approaches = ['Generic', 'Approach 1', 'Approach 2', 'Approach X']
# approaches = ['Initial sampling','Generic', 'Approach Prob','Approach Hybrid']
# approaches = ['Initial sampling','Generic_RVEA','Generic_IBEA']
# approaches = ['Initial sampling','Prob Old', 'Prob constant 1','Prob FE/FEmax']
# approaches = ['Generic', 'Approach Prob','Approach Hybrid']
# approaches = ['Generic', 'Probabilistic','Hybrid']
# approaches = ['7', '9', '11']
#approaches = ['Generic', 'TransferL', 'Probabilistic', 'Hybrid']
approaches = ['Gen','TL','Prob','Hyb']

nruns = 11
f_eval_limit = 40000
f_evals_per = 2000
f_iters = 20
pool_size = nruns

plot_boxplot = True

l = [approaches] * nruns
labels = [list(i) for i in zip(*l)]
labels = [y for x in labels for y in x]

p_vals_all = None
index_all = None


def arg_median(a):
    if len(a) % 2 == 1:
        return np.where(a == np.median(a))[0][0]
    else:
        l, r = len(a) // 2 - 1, len(a) // 2
        left = np.partition(a, l)[l]
        right = np.partition(a, r)[r]
        return [np.where(a == left)[0][0], np.where(a == right)[0][0]]


# calculating multivariate RMSE
def rmse_mv_calc(surrogate_objectives_nds,actual_objectives_nds):
    if np.size(surrogate_objectives_nds)==np.size(actual_objectives_nds):
        rmse_mv_sols = 0
        for i in range(np.shape(actual_objectives_nds)[0]):
            rmse_mv_sols += distance.euclidean(surrogate_objectives_nds[i, :obj], actual_objectives_nds[i, :])
        rmse_mv_sols = rmse_mv_sols / np.shape(actual_objectives_nds)[0]
        return rmse_mv_sols
    else:
        return -1000


for samp in sampling:
    for prob in problems:
        for obj in objectives:
            for n_vars in dims:
                fig = plt.figure(1, figsize=(6, 3.5))
                # fig = plt.figure()
                ax = fig.add_subplot(111)
                fig.set_size_inches(5.5, 4)

                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax = fig.add_subplot(111)
                # fig.set_size_inches(5, 5)
                # plt.xlim(0, 1)
                # plt.ylim(0, 1)

                # if save_fig == 'pdf':
                #    plt.rcParams["text.usetex"] = True
                # with open(pareto_front_directory + '/True_5000_' + prob + '_' + obj + '.txt') as csv_file:
                #    csv_reader = csv.reader(csv_file, delimiter=',')


                # path_to_file = pareto_front_directory + '/' + 'Pareto_Weld'
                # infile = open(path_to_file, 'rb')
                # pareto_front = pickle.load(infile)
                # infile.close()
                # problem_weld = WeldedBeam()
                # pareto_front = problem_weld.pareto_front()

                for algo in emo_algorithm:
                    # igd_all = np.zeros([nruns, np.shape(modes)[0]])
                    igd_all = None
                    solution_ratio_all = None
                    hv_progress = None
                    hv_progress_dict = {}
                    hv_progress_df = []
                    hv_df = pd.DataFrame()
                    for mode, mode_count in zip(modes, range(np.shape(modes)[0])):
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
                            soln_all = []
                            surr_all = []
                            pop_all = []
                            print("Reading...")
                            path_to_file1 = path_to_file + '/Run_' + str(run + 1) + '_soln_all'
                            with open(path_to_file1, 'r') as f:
                                reader = csv.reader(f)
                                for line in reader: soln_all.append(line)
                            path_to_file1 = path_to_file + '/Run_' + str(run + 1) + '_surr_all'
                            with open(path_to_file1, 'r') as f:
                                reader = csv.reader(f)
                                for line in reader: surr_all.append(line)
                            path_to_file1 = path_to_file + '/Run_' + str(run + 1) + '_pop_all'
                            with open(path_to_file1, 'r') as f:
                                reader = csv.reader(f)
                                for line in reader: pop_all.append(line)
                            soln_all = np.array(soln_all, dtype=np.float32)
                            surr_all = np.array(surr_all, dtype=np.float32)
                            pop_all = np.array(pop_all, dtype=np.float32)

                            print(np.shape(soln_all))
                            print(np.shape(surr_all))
                            print(np.shape(pop_all))
                            print("..........")
                            """
                            else:
                                path_to_file1 = path_to_file + '/Run_' + str(run + 1) + '_soln'
                                soln_all = []
                                with open(path_to_file1, 'r') as f:
                                    reader = csv.reader(f)
                                    for line in reader: soln_all.append(line)
                                soln_all = np.array(soln_all, dtype=np.float32)
                            """
                            # print(actual_objectives_nds)
                            rmse_iter = []

                            max_iter = np.min((np.shape(soln_all)[0], int(f_eval_limit)))
                            f_evals_per = int(np.shape(surr_all)[0] / f_iters)
                            # for f_iter in range(int(max_iter / f_evals_per)):
                            for f_iter in range(f_iters):
                                # soln_all_temp = soln_all[f_iter * f_evals_per:(f_iter + 1) * f_evals_per, :]
                                if f_iter == 0:
                                    soln_all_temp = soln_all[0:100, :]
                                    surr_all_temp = surr_all[0:100, :]
                                else:
                                    soln_all_temp = soln_all[f_iter * f_evals_per:(f_iter + 1) * f_evals_per, :]
                                    surr_all_temp = surr_all[f_iter * f_evals_per:(f_iter + 1) * f_evals_per, :]

                                rmse_temp = rmse_mv_calc(surr_all_temp,soln_all_temp)
                                if f_iter == 0:
                                    rmse_temp = 0
                                rmse_iter = np.append(rmse_iter, rmse_temp)
                            return rmse_iter


                        temp = Parallel(n_jobs=11)(delayed(parallel_execute)(run, path_to_file) for run in range(nruns))

                        #temp = None
                        #for run in range(nruns):
                        #    temp = np.append(temp, parallel_execute(run, path_to_file))

                        temp = np.asarray(temp)

                        if hv_progress is None:
                            hv_progress = [temp]
                        else:
                            hv_progress = np.vstack((hv_progress, [temp]))
                        # igd_temp = np.transpose(temp[:, 0])
                        for i in range(nruns):
                            # for j in range(int(f_eval_limit / f_evals_per)):
                            for j in range(f_iters):
                                hv_progress_df.append([j * f_evals_per, i, approaches[mode_count], temp[i, j]])

                    hv_df = pd.DataFrame(hv_progress_df, columns=['Evaluations', 'Runs', 'Approaches', 'RMSE'])
                    print(hv_df)
                    color_map = plt.cm.get_cmap('viridis')
                    color_map = color_map(np.linspace(0, 1, 5))
                    color_map = color_map[1:5]
                    ax = sns.lineplot(x="Evaluations", y="RMSE",
                                      hue="Approaches", style="Approaches",
                                      markers=True, dashes=False, data=hv_df,palette=color_map)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    box = ax.get_position()
                    # ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position

                    # Put a legend to the right side
                    # ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)

                    #ax.set_title(prob + ', ' + str(obj) + ' objs, '+sx)
                    handles, labels = ax.get_legend_handles_labels()
                    # ax.legend(handles=handles, labels=labels)
                    ax.legend(handles=handles[1:], labels=labels[1:], frameon=False, loc='upper center',
                              bbox_to_anchor=(0.5, 1.15), ncol=5)

                    fig = ax.get_figure()
                    fig.show()
                    filename_fig = main_directory + '/' 'RMSE_progress_' + samp + '_' + algo + '_' + prob + '_' + str(
                        obj) + '_' + str(n_vars)
                    if save_fig == 'png':
                        fig.savefig(filename_fig + '.png', bbox_inches='tight')
                    else:
                        fig.savefig(filename_fig + '.pdf', bbox_inches='tight')
                    ax.clear()

