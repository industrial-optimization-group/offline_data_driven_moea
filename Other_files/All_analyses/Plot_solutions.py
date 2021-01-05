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
from pymop.problems.welded_beam import WeldedBeam
from pymop.problems.truss2d import Truss2D
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from matplotlib import rc

#rc('font',**{'family':'serif','serif':['Helvetica']})
rc('text', usetex=False)
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
#objectives = [3,5,6,8,10]

#problems = ['DTLZ2']
#problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
#problems = ['P1','P2']
problems = ['P1']
#problems = ['WELDED_BEAM'] #dims=4
#problems = ['TRUSS2D'] #dims=3

#modes = [1, 2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2, 7 = Approach_Prob
#modes = [0,7,70,71]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
#modes = [0,1,2]
modes = [0,1,9,7,8]
#modes = [8]
mode_length = int(np.size(modes))
#sampling = ['BETA', 'MVNORM']
#sampling = ['LHS']
#sampling = ['BETA','OPTRAND','MVNORM']
#sampling = ['OPTRAND']
sampling = ['MVNORM']
#sampling = ['LHS', 'MVNORM']

#emo_algorithm = ['RVEA','IBEA']
emo_algorithm = ['RVEA']
#emo_algorithm = ['IBEA']
#emo_algorithm = ['NSGAIII']
#emo_algorithm = ['MODEL_CV']

#med_index=[8,8,5,7,5]
#med_index=[7,1,0,7,3]
med_index=[10,0,5,8,6]

#approaches = ['Generic', 'Approach 1', 'Approach 2', 'Approach 3']
#approaches = ['Generic', 'Approach 1', 'Approach 2', 'Approach X']
#approaches = ['Initial sampling','Generic', 'Approach Prob','Approach Hybrid']
#approaches = ['Initial sampling','Generic_RVEA','Generic_IBEA']
#approaches = ['Initial sampling','Prob Old', 'Prob constant 1','Prob FE/FEmax']
#approaches = ['Generic', 'Approach Prob','Approach Hybrid']
approaches = ['Generic', 'Probabilistic','Hybrid']
#approaches = ['7', '9', '11']
#"DTLZ6": {"2": [10, 10], "3": [10, 10, 10], "5": [10, 10, 10, 10, 10]},
#"DTLZ4": {"2": [4, 4], "3": [4, 4, 4], "5": [4, 4, 4, 4, 4]},
hv_ref = {"DTLZ2": {"2": [3, 3], "3": [3, 3, 3], "5": [3, 3, 3, 3, 3]},
          "DTLZ4": {"2": [3, 3.1], "3": [3, 3, 3], "5": [3, 3, 3, 3, 3]},
          "DTLZ5": {"2": [2.5, 3], "3": [2.5, 3, 3], "5": [2, 2, 2, 2, 2]},
          "DTLZ6": {"2": [10, 10], "3": [10, 10, 10], "5": [7, 7, 7, 7, 7]},
          "DTLZ7": {"2": [1, 20], "3": [1, 1, 30], "5": [1, 1, 1, 1, 45]}}


nruns = 1
pool_size = nruns

plot_boxplot = True

l = [approaches]*nruns
labels = [list(i) for i in zip(*l)]
labels = [y for x in labels for y in x]

p_vals_all = None
index_all = None

for samp in sampling:
    for prob in problems:
        for obj in objectives:
            for n_vars in dims:

                fig = plt.figure(figsize=plt.figaspect(0.25))
                fig.set_size_inches(30, 5)
                #plt.xlim(0, 1)
                #plt.ylim(0, 1)
                #ax = fig.add_subplot(111, projection='3d')


                if problem_testbench is 'DTLZ':
                    pareto_front = np.genfromtxt(pareto_front_directory + '/True_5000_' + prob + '_' + str(obj) + '.txt'
                                             , delimiter=',')

                for algo in emo_algorithm:
                    #igd_all = np.zeros([nruns, np.shape(modes)[0]])
                    igd_all = None
                    solution_ratio_all = None
                    for mode, mode_count in zip(modes,range(np.shape(modes)[0])):
                        if obj==2:
                            ax = fig.add_subplot(1,np.shape(modes)[0],mode_count+1)
                        else:
                            ax = fig.add_subplot(1, np.shape(modes)[0], mode_count + 1,projection='3d')

                        ax.set_xlabel('$f_1$')
                        ax.set_ylabel('$f_2$')
                        ax.set_xlim([0, 0.5])
                        ax.set_ylim([0, 0.5])
                        if obj==3:
                            ax.set_zlabel('$f_3$')
                            ax.set_xlim([0, 1.6])
                            ax.set_ylim([0, 1.6])
                            ax.set_zlim([0, 1.6])
                        if problem_testbench is 'DTLZ':
                            path_to_file = main_directory \
                                       + '/Offline_Mode_' + str(mode) + '_' + algo + \
                                       '/' + samp + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                        else:
                            path_to_file = main_directory \
                                           + '/Offline_Mode_' + str(mode) + '_' + algo + \
                                           '/' + samp + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                        print(path_to_file)


                        def parallel_execute(run, path_to_file):
                            run=med_index[mode_count]
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
                                """
                                if problem_testbench is 'DDMOPP':
                                    path_to_file = path_to_file + '/Run_' + str(run) + '_SOLN'
                                    infile = open(path_to_file, 'rb')
                                    results_data=pickle.load(infile)
                                    infile.close()
                                    if mode == 0:
                                        actual_objectives_nds = results_data["actual_objectives_nds"]
                                    else:
                                        actual_objectives_nds = results_data["obj_solns"]
                                    actual_objectives_nds = np.array(actual_objectives_nds, dtype=np.float32)
                                    ref = [1]*obj
                                else:
                                    path_to_file = path_to_file + '/Run_' + str(run) + '_NDS'
                                    infile = open(path_to_file, 'rb')
                                    results_data = pickle.load(infile)
                                    infile.close()
                                    actual_objectives_nds = results_data["actual_objectives_nds"]
                                    #non_dom_surr = results_data["surrogate_objectives_nds"]

                                    ref = hv_ref[prob][str(obj)]
                                """
                                if problem_testbench is 'DDMOPP':
                                    if mode == 9:
                                        path_to_file = path_to_file + '/Run_' + str(run + 1) + '_soln'
                                        actual_objectives_nds = []
                                        with open(path_to_file, 'r') as f:
                                            reader = csv.reader(f)
                                            for line in reader: actual_objectives_nds.append(line)

                                    else:
                                        path_to_file = path_to_file + '/Run_' + str(run) + '_SOLN'
                                        infile = open(path_to_file, 'rb')
                                        results_data = pickle.load(infile)
                                        infile.close()
                                        if mode == 0:
                                            actual_objectives_nds = results_data["actual_objectives_nds"]
                                        else:
                                            actual_objectives_nds = results_data["obj_solns"]
                                    actual_objectives_nds = np.array(actual_objectives_nds, dtype=np.float32)
                                    ref = [2] * obj

                                else:
                                    if mode == 9:
                                        path_to_file = path_to_file + '/Run_' + str(run + 1) + '_soln'
                                        actual_objectives_nds = []
                                        with open(path_to_file, 'r') as f:
                                            reader = csv.reader(f)
                                            for line in reader: actual_objectives_nds.append(line)
                                        actual_objectives_nds = np.array(actual_objectives_nds, dtype=np.float32)
                                    else:
                                        path_to_file = path_to_file + '/Run_' + str(run) + '_NDS'
                                        infile = open(path_to_file, 'rb')
                                        results_data = pickle.load(infile)
                                        infile.close()
                                        actual_objectives_nds = results_data["actual_objectives_nds"]
                                        # non_dom_surr = results_data["surrogate_objectives_nds"]

                                    ref = hv_ref[prob][str(obj)]
                                #print(actual_objectives_nds)
                                if np.shape(actual_objectives_nds)[0] > 1:
                                    non_dom_front = ndx(actual_objectives_nds)
                                    actual_objectives_nds = actual_objectives_nds[non_dom_front[0][0]]
                                else:
                                    actual_objectives_nds = actual_objectives_nds.reshape(1, obj)
                                #print(np.shape(actual_objectives_nds))
                                solution_ratio = 0
                                hyp = hv(actual_objectives_nds)
                                hv_x = hyp.compute(ref)
                                print(np.amax(actual_objectives_nds,axis=0))
                                if obj==2:
                                    ax.scatter(actual_objectives_nds[:, 0], actual_objectives_nds[:, 1])
                                else:
                                    ax.scatter(actual_objectives_nds[:, 0], actual_objectives_nds[:, 1],actual_objectives_nds[:, 2])


                                #return [hv_x, solution_ratio]


                        #temp = Parallel(n_jobs=11)(delayed(parallel_execute)(run, path_to_file) for run in range(nruns))
                        #temp=None
                        for run in range(nruns):
                            parallel_execute(run+7, path_to_file)
                    if obj==2:
                        filename_fig = main_directory + '/2d_scatter_' + samp + '_' + algo + '_' + prob + '_' + str(
                        obj) + '_' + str(n_vars) + '.pdf'
                    else:
                        filename_fig = main_directory + '/3d_scatter_' + samp + '_' + algo + '_' + prob + '_' + str(
                        obj) + '_' + str(n_vars) + '.pdf'
                    print(filename_fig)
                    fig.savefig(filename_fig, bbox_inches='tight')
                    plt.show()
