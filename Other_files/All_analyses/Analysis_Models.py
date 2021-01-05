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

pareto_front_directory = 'True_Pareto_5000'


dims = 10
#dims = 4


main_directory = 'New_tests_offline'
#main_directory = 'Tests_Final'
#main_directory = 'Tests_new_adapt'
#main_directory = 'Tests_toys'

#objectives = [2, 3, 5]
objectives = [2,3]
#objectives = [5]

problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
#problems = ['DTLZ5','DTLZ6','DTLZ7']
#problems = ['DTLZ2']
#problems = ['WELDED_BEAM']
#problems = ['TRUSS2D']

#modes = [1, 2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
#modes = [2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
modes = [1]

sampling = ['BETA', 'MVNORM']
#sampling = ['OPTRAND']
#sampling = ['DIRECT']
#sampling = ['MVNORM']

#emo_algorithm = ['RVEA','IBEA']
#emo_algorithm = ['RVEA']
#emo_algorithm = ['IBEA']
emo_algorithm = ['MODEL_CV']


nruns = 11
pool_size = nruns

plot_boxplot = True


for obj in objectives:
    for prob in problems:
        fig = plt.figure(1, figsize=(9, 6))
        ax = fig.add_subplot(111)
        for algo in emo_algorithm:
            mean_r_sq_cv_all = None
            mean_rmse_cv_all = None
            for samp in sampling:
                #igd_all = np.zeros([nruns, np.shape(modes)[0]])
                 for mode, mode_count in zip(modes,range(np.shape(modes)[0])):
                    path_to_file = main_directory \
                                   + '/Offline_Mode_' + str(mode) + '_' + algo + \
                                   '/' + samp + '/' + prob + '_' + str(obj)
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
                        path_to_file = path_to_file + '/Run_' + str(run)
                        infile = open(path_to_file, 'rb')
                        results_data=pickle.load(infile)
                        infile.close()
                        #mean_r_sq_cv = np.asarray(results_data["mean_r_squared"])
                        #mean_rmse_cv = np.asarray(results_data["mean_rmse"])
                        #return [mean_r_sq_cv[:, 0], mean_rmse_cv[:, 0]]
                        r_sq_cv = np.asarray(results_data["r_squared"])
                        rmse_cv = np.asarray(results_data["rmse"])
                        return [np.median(r_sq_cv,axis=1), np.median(rmse_cv,axis=1)]


                    temp = Parallel(n_jobs=11)(delayed(parallel_execute)(run, path_to_file) for run in range(nruns))
                    temp = np.asarray(temp)
                    mean_r_sq_cv_temp = temp[:][:, 0]
                    mean_rmse_cv_temp = temp[:][:, 1]
                    #print(mean_r_sq_cv_temp)
                    #for run in range(nruns):
                    #    parallel_execute(run, path_to_file)

                    if plot_boxplot is True:
                        if mean_r_sq_cv_all is None:
                            mean_r_sq_cv_all = mean_r_sq_cv_temp
                            mean_rmse_cv_all = mean_rmse_cv_temp
                        else:
                            mean_r_sq_cv_all = np.hstack((mean_r_sq_cv_all , mean_r_sq_cv_temp))
                            mean_rmse_cv_all = np.hstack((mean_rmse_cv_all , mean_rmse_cv_temp))


            print(mean_r_sq_cv_all)
            count=0
            samp_size = len(sampling)
            obj_samp = []
            mean_r_sq_cv_all_temp = np.zeros((11,obj*samp_size))
            for i in range(obj):
                for j in range(samp_size):
                    mean_r_sq_cv_all_temp[:,count] = mean_r_sq_cv_all[:, i+j*samp_size]
                    count = count+1
                    obj_samp.append(sampling[j]+'_Obj'+str(i))
            mean_r_sq_cv_all = mean_r_sq_cv_all_temp
            bp = ax.boxplot(mean_r_sq_cv_all, showfliers=False)
            ax.set_title('R_squared_' + prob + '_' + str(obj))
            ax.set_xlabel('Objectives')
            ax.set_ylabel('R_sq')
            ax.set_xticklabels(obj_samp, rotation=45, fontsize=8)
            filename_fig = main_directory + '/R_SQ_Med' + '_' + prob + '_' + str(obj) + '.png'
            fig.savefig(filename_fig, bbox_inches='tight')
            ax.clear()
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