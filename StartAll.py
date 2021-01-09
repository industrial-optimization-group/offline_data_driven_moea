#import Main_Execute_Prob as mexeprob
import Main_Execute as mexe
#import Main_Execute_interactive as mexe_int
import pickle
import os
from joblib import Parallel, delayed
import datetime
#import os
#os.environ["OMP_NUM_THREADS"] = "1"
#import Telegram_bot.telegram_bot_messenger as tgm
#dims = [5,8,10] #,8]
dims = [10]
#dims = [27]

#sample_size = 1000
sample_size = 109
#dims = 4
############################################
folder_data = 'AM_Samples_109_Final'
#folder_data = 'AM_Samples_1000'

#problem_testbench = 'DTLZ'
problem_testbench = 'DDMOPP'
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
main_directory = 'Tests_CSC_4'
#main_directory = 'Tests_additional_obj1'
#main_directory = 'Tests_Interactive_2'
#main_directory = 'Tests_new_adapt'
#main_directory = 'Tests_toys'

#objectives = [4,6]#,5]
#objectives = [11]
objectives = [2]
#objectives = [2,3,5]
#objectives = [2,3,4,5,6,8,10]
#objectives = [3,5,6,8,10]
#objectives = [3,5,6,8,10]

#problems = ['DTLZ5']
#problems = ['GAA']
#problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
#problems = ['P1','P2']
problems = ['P1']
#problems = ['WELDED_BEAM'] #dims=4
#problems = ['TRUSS2D'] #dims=3


#modes = [1, 7, 8]  # 1 = Gen-RVEA, 7 = Prob-RVEA 1 , 8 = Hyb-RVEA
#modes = [12, 72, 82] # 12 = Gen-MOEA/D, 72 = Prob-MOEA/D, 82 = Hyb-MOEA/D

modes = [82]


sampling = ['LHS']
#sampling = ['MVNORM']
#sampling = ['LHS', 'MVNORM']

emo_algorithm = ['RVEA']
#emo_algorithm = ['IBEA']
#emo_algorithm = ['NSGAIII']
#emo_algorithm = ['MODEL_CV']

interactive = False

#############################################


nruns = 1
n_parallel_jobs = 1
log_time = str(datetime.datetime.now())
#tgm.send(msg='Started testing @'+str(log_time))



def parallel_execute(run, mode, algo, prob, n_vars, obj, samp):
    path_to_file = main_directory \
                + '/Offline_Mode_' + str(mode) + '_' + algo + \
                '/' + samp + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
    print(path_to_file+'__Run__'+str(run))
    with open(main_directory+"/log_"+log_time+".txt", "a") as text_file:
        text_file.write("\n"+path_to_file+"___"+str(run)+"_____"+str(datetime.datetime.now()))
    if not os.path.exists(path_to_file):
        os.makedirs(path_to_file)
    results_dict = mexe.run_optimizer(problem_testbench=problem_testbench, 
                                        folder_data=folder_data,
                                        problem_name=prob,
                                        nobjs=obj,
                                        nvars=n_vars,
                                        sampling=samp,
                                        is_data=True,
                                        run=run,
                                        approach=mode)
    path_to_file = path_to_file + '/Run_' + str(run)
    outfile = open(path_to_file, 'wb')
    pickle.dump(results_dict, outfile)
    outfile.close()


try:
    
    temp = Parallel(n_jobs=n_parallel_jobs)(
        delayed(parallel_execute)(run, mode, algo, prob, n_vars, obj, samp) for run in range(nruns) 
        for algo in emo_algorithm
        for prob in problems
        for n_vars in dims
        for samp in sampling
        for obj in objectives
        for mode in modes)
#    for run in range(nruns):
#        parallel_execute(run, path_to_file)
#    tgm.send(msg='Finished Testing: \n' + path_to_file)
except Exception as e:
#    tgm.send(msg='Error occurred : \n' + path_to_file + '\n' + str(e))
    print(e)
#for run in range(nruns):
#    parallel_execute(run, path_to_file)
#tgm.send(msg='All tests completed successfully')

