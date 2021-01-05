import Probability_wrong as pwrong
import pickle

path_to_file = 'data_apd_dist_plt'
infile = open(path_to_file, 'rb')
dist_obj = pickle.load(infile)
infile.close()
dist_obj['pwrng'].plt_density(dist_obj['samples'])
print(dist_obj)

