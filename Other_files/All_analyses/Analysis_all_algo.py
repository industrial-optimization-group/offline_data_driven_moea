import numpy as np
import pickle
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import csv
from IGD_calc import igd
from pymop.problems.welded_beam import WeldedBeam
from pymop.problems.truss2d import Truss2D

pareto_front_directory = 'True_Pareto_5000'


#dims = 10
dims = 4

#main_directory = 'New_tests_offline'
#main_directory = 'Tests_Final'
main_directory = 'Tests_toys'

#objectives = [2, 3, 5]
#objectives = [3,5]
objectives = [2]

#problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
#problems = ['DTLZ2','DTLZ6','DTLZ7']
#problems = ['DTLZ7']
problems = ['WELDED_BEAM']
#problems = ['TRUSS2D']

modes = [1, 2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
#modes = [2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
#modes = [3]

#sampling = ['LHS', 'OPTRAND', 'BETA']
sampling = ['LHS']
#sampling = ['DIRECT']

emo_algorithm = ['RVEA','IBEA']
#emo_algorithm = ['IBEA']
#emo_algorithm = ['IBEA']
nruns = 11
pool_size = nruns

plot_boxplot = True

for samp in sampling:
    for obj in objectives:
        for prob in problems:
            fig = plt.figure(1, figsize=(9, 6))
            ax = fig.add_subplot(111)
            #with open(pareto_front_directory + '/True_5000_' + prob + '_' + obj + '.txt') as csv_file:
            #    csv_reader = csv.reader(csv_file, delimiter=',')
            #pareto_front = np.genfromtxt(pareto_front_directory + '/True_5000_' + prob + '_' + str(obj) + '.txt'
            #                             , delimiter=',')
            #problem_weld = WeldedBeam()
            #pareto_front = problem_weld.pareto_front()
            path_to_file = pareto_front_directory + '/' + 'Pareto_Weld'
            infile = open(path_to_file, 'rb')
            pareto_front = pickle.load(infile)
            infile.close()

            for mode in modes:
                #igd_all = np.zeros([nruns, np.shape(modes)[0]])
                igd_all = None
                solution_ratio_all = None
                for algo, algo_count in zip(emo_algorithm, range(np.shape(emo_algorithm)[0])):
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
                        path_to_file = path_to_file + '/Run_' + str(run) + '_NDS'
                        infile = open(path_to_file, 'rb')
                        results_data=pickle.load(infile)
                        infile.close()
                        non_dom_front = results_data["actual_objectives_nds"]
                        non_dom_surr = results_data["surrogate_objectives_nds"]
                        solution_ratio = np.shape(non_dom_front)[0]/np.shape(non_dom_surr)[0]
                        #print(non_dom_front)
                        #plt.scatter(non_dom_front[:,0],non_dom_front[:,1])
                        #plt.show()
                        #print(solution_ratio)
                        return [igd(pareto_front,non_dom_front), solution_ratio]
                        #zz = np.amax(non_dom_front,axis=0)
                        #return zz


                    temp = Parallel(n_jobs=11)(delayed(parallel_execute)(run, path_to_file) for run in range(nruns))
                    temp=np.asarray(temp)
                    igd_temp = np.transpose(temp[:, 0])
                    solution_ratio_temp = np.transpose(temp[:, 1])
                    #for run in range(nruns):
                    #    parallel_execute(run, path_to_file)

                    if plot_boxplot is True:
                        if igd_all is None:
                            igd_all = igd_temp
                            solution_ratio_all = solution_ratio_temp
                        else:
                            igd_all = np.vstack((igd_all, igd_temp))
                            solution_ratio_all = np.vstack((solution_ratio_all,solution_ratio_temp))
                igd_all = np.transpose(igd_all)
                solution_ratio_all = np.transpose(solution_ratio_all)
                #print(np.max(igd_all))
                #print(np.max(solution_ratio_all))
                #print(igd_all)
                #print(solution_ratio_all)
                bp = ax.boxplot(igd_all) #, showfliers=False)
                ax.set_title('IGD_'+ samp + '_Mode_' + str(mode) + '_' + prob + '_' + str(obj))
                ax.set_xlabel('Algorithms')
                ax.set_ylabel('IGD')
                ax.set_xticklabels(emo_algorithm, rotation=45, fontsize=8)
                filename_fig = main_directory + '/IGD_'+ samp + '_Mode_' + str(mode) + '_' + prob + '_' + str(obj) + '.png'
                fig.savefig(filename_fig, bbox_inches='tight')
                ax.clear()
                #bp = ax.boxplot(solution_ratio_all, showfliers=False)
                #filename_fig = main_directory + '/SolnRatio_' + samp + '_Mode_' + str(mode) + '_' + prob + '_' + str(obj) + '.png'
                #fig.savefig(filename_fig, bbox_inches='tight')
                #ax.clear()