#import Main_Execute_Prob as mexeprob
import Main_Execute as mexe
#import Main_Execute_interactive as mexe_int
import pickle
import pickle_to_mat_converter as pickmat
import os
from joblib import Parallel, delayed
import datetime

#convert_to_mat = True
convert_to_mat = False
#import Telegram_bot.telegram_bot_messenger as tgm
#dims = [5,8,10] #,8]
dims = [10]
#dims = [27]

#sample_size = 2000
sample_size = 2000
#dims = 4
############################################
#folder_data = 'AM_Samples_109_Final'
#folder_data = 'AM_Samples_1000'

problem_testbench = 'DTLZ'
#problem_testbench = 'DDMOPP'
#problem_testbench = 'GAA'
"""
objs(1) = max_NOISE;
objs(2) = max_WEMP;
objs(3) = max_DOC;
objs(4) = max_ROUGH;
objs(5) = max_WFUEL;
objs(6) = max_PURCH;
objs(7) = -min_RANGE;
objs(8) = -min_LDMAX;
objs(9) = -min_VCMAX;
objs(10) = PFPF;
"""

#main_directory = 'Offline_Prob_DDMOPP3'
#main_directory = 'Tests_Probabilistic_Finalx_new'
#main_directory = 'Tests_additional_obj1'
#main_directory = 'Tests_Gpy_1'
#main_directory = 'Tests_new_adapt'
#main_directory = 'Tests_toys'
#main_directory = 'Test_Gpy3'
main_directory = 'Test_DR_4'  #DR = Datatset Reduction
#main_directory = 'Test_RF'

objectives = [3,5,7]
#objectives = [5]
#objectives = [3,5,7]
#objectives = [2,3,5]
#objectives = [2,3,4,5,6,8,10]
#objectives = [3,5,6,8,10]
#objectives = [3,5,6,8,10]

#problems = ['DTLZ2']
#problems = ['GAA']
problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
#problems = ['P1','P2']
#problems = ['P2']
#problems = ['WELDED_BEAM'] #dims=4
#problems = ['TRUSS2D'] #dims=3

#modes = [1, 2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2, 7 = Approach_Prob
#modes = [7]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
#modes = [1, 7, 8]
#approaches = ["generic_fullgp","generic_sparsegp"]
#approaches = ["generic_fullgp","generic_sparsegp","strategy_1"]
#approaches = ["generic_fullgp"]
#approaches = ["generic_sparsegp"]
#approaches = ["strategy_1"]
#approaches = ["strategy_2"]
#approaches = ["strategy_3"]
#approaches = ["rf"]
#approaches = ["htgp"]
#approaches = ["generic_fullgp","htgp"]
approaches = ["generic_fullgp","generic_sparsegp","htgp"]
#approaches = ["generic_sparsegp"]

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

interactive = False

#############################################


nruns = 11
log_time = str(datetime.datetime.now())
#tgm.send(msg='Started testing @'+str(log_time))

for samp in sampling:
    for obj in objectives:
        for n_vars in dims:
            for prob in problems:
                for algo in emo_algorithm:
                    for approach in approaches:
                        path_to_file = './data/test_runs/' + main_directory \
                                       + '/Offline_Mode_' + approach + '_' + algo + \
                                       '/' + samp + '/' + str(sample_size) + '/' + problem_testbench  + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                        print(path_to_file)
                        with open('./data/test_runs/'+main_directory+"/log_"+log_time+".txt", "a") as text_file:
                            text_file.write("\n"+path_to_file+"_____"+str(datetime.datetime.now()))
                        if not os.path.exists(path_to_file):
                            os.makedirs(path_to_file)
                            print("Creating Directory...")

                        def parallel_execute(run, path_to_file):
                            if convert_to_mat is False:
                                results_dict = mexe.run_optimizer(problem_testbench=problem_testbench, 
                                                                    problem_name=prob, 
                                                                    nobjs=obj, 
                                                                    nvars=n_vars, 
                                                                    sampling=samp, 
                                                                    nsamples=sample_size, 
                                                                    is_data=True, 
                                                                    surrogate_type=approach,
                                                                    run=run)
                                path_to_file = path_to_file + '/Run_' + str(run)
                                outfile = open(path_to_file, 'wb')
                                pickle.dump(results_dict, outfile)
                                outfile.close()
                            else:
                                path_to_file = path_to_file + '/Run_' + str(run)
                                pickmat.convert(path_to_file, path_to_file+'.mat')



                        try:
                            temp = Parallel(n_jobs=nruns)(
                                delayed(parallel_execute)(run, path_to_file) for run in range(nruns))
                        #    for run in range(nruns):
                        #        parallel_execute(run, path_to_file)
                        #   tgm.send(msg='Finished Testing: \n' + path_to_file)
                        except Exception as e:
                        #    tgm.send(msg='Error occurred : \n' + path_to_file + '\n' + str(e))
                            print(e)        
                        #for run in range(nruns):
                        #    parallel_execute(run, path_to_file)
#tgm.send(msg='All tests completed successfully')

