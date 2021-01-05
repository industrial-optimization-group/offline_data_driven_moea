from plotly.offline.offline import plot
import plotly_express as ex
import pandas as pd
import numpy as np

def plot_vals(objs, unc, preference, iteration, interaction_count, ideal, nadir, min, max):
    objs_orig = objs
    columns = ["f_"+str(i+1) for i in range(np.shape(objs)[1])]
    range_plot = np.vstack((ideal,nadir))
    range_plot = np.hstack((range_plot,[[3],[3]]))
    if np.shape(objs)[0] > 0:
        unc_avg = np.mean(unc, axis=1)
        #unc_avg = (unc_avg-np.min(unc_avg))/(np.max(unc_avg)-np.min(unc_avg))
        unc_avg = (unc_avg - min) / (max-min)
        objs_col = unc_avg.reshape(-1, 1)
        objs = np.hstack((objs, objs_col))
    objs = np.vstack((objs, range_plot))
    objs = pd.DataFrame(objs, columns=columns + ["color"])
    if preference is not None:
        pref = pd.DataFrame(np.hstack((preference.reshape(1,-1), [[2]])), columns=columns + ["color"])
        data_final = pd.concat([objs, pref])
    else:
        data_final = objs
    #color_scale_custom= [(0.0,'rgb(0,0,0)'),(0.5,'rgb(160,90,0)'),(0.5,'green'),(0.75,'green'),(0.75,'blue'),(1.0,'blue')]
    #color_scale_custom = [(0.0, 'rgb(36,86,104)'), (0.5, 'rgb(237,239,93)'), (0.5, 'red'), (0.75, 'red'), (0.75, 'lightgray'),
    #                      (1.0, 'lightgray')]
#    color_scale_custom = [(0.0, 'rgb(69,2,86)'), (0.5, 'rgb(249,231,33)'), (0.5, 'red'), (0.75, 'red'), (0.75, 'white'),
#                          (1.0, 'white')]
#    color_scale_custom = [(0.0, 'rgb(69,2,86)'), (0.125, 'rgb(59,28,140)'), (0.25, 'rgb(33,144,141)'),
#                          (0.375, 'rgb(90,200,101)'), (0.5, 'rgb(249,231,33)'),
#                          (0.5, 'red'), (0.75, 'red'), (0.75, 'white'),
#                          (1.0, 'white')]
    color_scale_custom = [(0.0, 'rgb(69,2,86)'), (0.083, 'rgb(59,28,140)'), (0.167, 'rgb(33,144,141)'),
                          (0.25, 'rgb(90,200,101)'), (0.334, 'rgb(249,231,33)'),
                          (0.334, 'red'), (0.7, 'red'), (0.7, 'white'),
                          (1.0, 'white')]
    #color_scale_custom = ['Inferno', (1.5, 'green'), (2.5, 'green'), (2.5, 'white'),
    #                      (3.5, 'white')]
    #color_scale_custom = [(0.0,'red'),(1.5,'red'),(1.5, 'green'), (2.5, 'green'), (2.5, 'white'),
    #                      (3.5, 'white')]
    fig = ex.parallel_coordinates(
            data_final,
            dimensions=columns,
            color="color", color_continuous_scale=color_scale_custom, range_color=(0,3))
    plot(fig, filename="testfile_"+str(iteration)+"_"+str(interaction_count)+".html")
    print('Plotting done!!')
