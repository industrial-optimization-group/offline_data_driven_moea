import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import csv
import matplotlib.colors as matcol
from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Helvetica']})
#rc('text', usetex=True)
#plt.rcParams.update({'font.size': 15})


#comparison = 'HV'
comparison = 'RMSE'
summary = []
#approaches = ['Init','Gen','TL','Prob','Hyb']
approaches = ['Gen','TL','Prob','Hyb']
path_to_file = "HV_heatmap.csv"
path_to_file2 = "RMSE_heatmap.csv"
if comparison is 'HV':
    with open(path_to_file, 'r') as f:
        reader = csv.reader(f)
        for line in reader: summary.append(line)
else:
    with open(path_to_file2, 'r') as f:
        reader = csv.reader(f)
        for line in reader: summary.append(line)

summary_df = []
for i in range(np.shape(summary)[0]):
    for j in range(len(approaches)):
        if comparison is 'HV':
            summary_df.append([summary[i][0]+'_'+summary[i][2]+'_'+summary[i][1]+'_'+summary[i][3],approaches[j],int(summary[i][j+4])])
        else:
            summary_df.append(
                [summary[i][0] + '_' + summary[i][2] + '_' + summary[i][1] + '_' + summary[i][3], approaches[j],
                 int(summary[i][j + 4])])

fig = plt.figure(1, figsize=(2.7, 5))
# fig = plt.figure()
ax = fig.add_subplot(111)
summary_df2 = pd.DataFrame(summary_df, columns=['Instances','Approaches','Rank'])
summary_df2 = summary_df2.pivot("Instances", "Approaches", "Rank")
#summary_df2 = summary_df2.reindex(summary_df2.sort_values(by='Instances', ascending=True).index)
if comparison is 'HV':
    summary_df2 = summary_df2[['Init','Gen', 'TL', 'Prob', 'Hyb']]
else:
    summary_df2=summary_df2[['Gen','TL','Prob','Hyb']]
print(summary_df2)
color_map = plt.cm.get_cmap('viridis')
reversed_color_map = color_map.reversed()
reversed_color_map = reversed_color_map(np.linspace(0,1,(len(approaches))))
reversed_color_map = matcol.ListedColormap(reversed_color_map)
ax = sns.heatmap(summary_df2, yticklabels=False, cmap=reversed_color_map,cbar_kws = {'orientation':'horizontal'})
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([1, len(approaches)])
colorbar.set_ticklabels(['Best', 'Worst'])
fig = ax.get_figure()
#fig.show()
filename_fig = 'Heatmap_'+comparison
fig.savefig(filename_fig + '.pdf', bbox_inches='tight')