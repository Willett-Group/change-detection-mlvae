import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# names of 'run' folders to plot
names = [1,2,3,4,6]

diffs = []
means = []
stds = []

lower = []
median = []
upper = []

for d in names:
    run = '0'+str(d) if d < 10 else str(d)
    path = ''

    for dir in os.listdir(os.path.join('sqerrors', 'run'+run)):
        if dir[-4:] == '.txt':
            path = os.path.join('sqerrors', 'run'+run, dir)
    if path == '':
        raise Exception('txt file missing')

    with open(path) as f:
        cps_hat = f.readline().split(' ')[0:-1]
        cps_hat = np.array([int(i) for i in cps_hat])
        cps = f.readline().split(' ')[0:-1]
        cps = np.array([int(i) for i in cps])
        diff = np.abs(cps_hat - cps)
        
        lower.append(np.percentile(diff, [20]))
        median.append(np.percentile(diff, [50]))
        upper.append(np.percentile(diff, [80]))
        
        diffs.append(diff)
        means.append(np.sum(diff) / len(diff))
        stds.append(np.std(diff))

        # plot histograms
        '''
        plt.hist(diff, [0,2,5,10,20], weights=np.ones(len(diff)) / len(diff))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel('|eta - eta_hat|')
        plt.ylabel('percentage of test samples')
        plt.title('histogram for ' + names[d])
        plt.show()
        plt.close()
        '''

# plot the average of |eta - eta_hat| across test samples
ticks = ['1','0.75','0.5','0.25','Unif(0,1)']
plt.errorbar(ticks, means, yerr=stds, fmt='o')
plt.xticks(ticks, ticks, fontsize=7)
plt.xlabel('theta')
plt.ylabel('mean |eta-eta_hat| across all test samples')
plt.title('mean |eta-eta_hat| (784-d original setting)')
plt.show()
plt.close()


# plot the percentiles of |eta-eta_hat| across test samples
# lower, median, upper = np.percentile(diffs, [20,50,80], axis=1)
plt.plot(ticks, median)
plt.plot(ticks, lower, c='red')
plt.plot(ticks, upper)
plt.scatter(ticks, median, s=10)
plt.scatter(ticks, lower, c='red', s=10)
plt.scatter(ticks, upper, s=10)
plt.xticks(ticks, ticks, fontsize=7)
plt.xlabel('theta')
plt.ylabel('|eta-eta_hat|')
plt.title('20,50,80 percentiles of |eta-eta_hat| across all test samples')
plt.show()
plt.close()


'''
# plot error bars aross n's
avg = np.average(diffs, axis=0)
std = np.std(diffs, axis=0)

plt.errorbar(range(len(cps)), avg, yerr=std, fmt='o')
plt.xlabel('test samples X1-X100')
plt.ylabel('average |eta - eta_hat| across different n values')
plt.title('n =  T=500')
plt.show()
plt.close()
'''