import desdeo_emo.othertools.ProbabilityWrong as pwrong
import pickle

#path_to_file = 'data_apd_dist_plt'
path_to_file = 'data_pbi_dist_SF_current'
path_to_file2 = 'data_pbi_dist_SF_current_samples'
infile = open(path_to_file, 'rb')
infile2 = open(path_to_file2, 'rb')
dist_obj = pickle.load(infile)
infile.close()
dist_SF = pickle.load(infile2)
infile2.close()
dist_obj.plt_density(dist_SF)
#pwrong_offspring.plt_density(values_SF_offspring.reshape(20,1,1000))
print(dist_SF)

