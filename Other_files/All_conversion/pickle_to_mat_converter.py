import numpy, scipy.io
import pickle
 
def convert(source_name, dest_name):
    a=pickle.load(open(source_name, "rb"))
    scipy.io.savemat(dest_name, mdict={'run_data':a})
