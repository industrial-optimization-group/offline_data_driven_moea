import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
# Create data
x = np.random.random((5,5))*40000
x[0][0]=0
x[0][1]=40000
df = pd.DataFrame(x, columns=["a","b","c","d","e"])
# Define two rows for subplots
fig, (cax, ax) = plt.subplots(nrows=2, figsize=(4,3),  gridspec_kw={"height_ratios":[0.1, 1]})
# Draw heatmap
sns.heatmap(df, ax=ax, cbar=False,cmap="viridis")
# colorbar
fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")
plt.show()
fig.savefig('color_bar.pdf', bbox_inches='tight')