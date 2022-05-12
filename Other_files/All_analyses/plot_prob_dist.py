import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import scipy.stats as stats
import math

rc('font',**{'family':'serif','serif':['serif']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 17})

mu1 = 3
variance1 = 1
mu2 = 5
variance2 = 0.5
t=0.1

sigma1 = math.sqrt(variance1)
sigma2 = math.sqrt(variance2)
x = np.linspace(-1, 9, 10000)
y = np.linspace(0,0.6,1000)
fig, ax = plt.subplots()
plt.ylim(0,0.65)
plt.xlim(-1,9)
ax.set_xlabel('Objective value')
ax.set_ylabel('Probability density')
plt.plot(np.ones(1000)*mu1, y, ':b')
plt.plot(np.ones(1000)*mu2, y, ':g')
ax.annotate('$\mu_A$',(mu1-t,0.61))
ax.annotate('$\mu_B$',(mu2-t,0.61))

plt.plot(x, stats.norm.pdf(x, mu1, sigma1), 'b', label='PDF$_A$')
plt.plot(x, stats.norm.pdf(x, mu2, sigma2), 'g', label='PDF$_B$')
ax.legend()
x_pnt=3.7
plt.scatter(x_pnt,stats.norm.pdf(x_pnt, mu1, sigma1),c='r', s=80, marker='*')
plt.plot(np.ones(1000)*x_pnt, y, '-.r')
ax.annotate('$y$',(x_pnt-t,0.61))

x_cdf = np.linspace(0,x_pnt,1000)
y_B=stats.norm.pdf(x_cdf, mu2, sigma2)

ax.fill_between(x_cdf,y_B , 0, facecolor='cyan',alpha=0.5)
#plt.show()
fig.savefig('Thesis_pdf_A_B.pdf')