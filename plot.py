import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

config = {
    'experiment_setting': 'clevr_change',
    'model': 'dfcvae', # 'dfcvae', 'linearvae', 'convvae', 'resnetvae'
}

root_dir = 'experiments/' + config['experiment_setting'] + '/' + config['model']
title = config['experiment_setting'] + ' ' + config['model']

ticks = []
diffs, means, stds, lower, median, upper = [],[],[],[],[],[]

for dir in os.listdir(root_dir):
    if dir.isdigit() and int(dir) not in [33]:
        with open(os.path.join(root_dir, dir, 'config.txt'), 'r') as f1:
            l1, l2 = f1.read().splitlines()[-2:]
            tick = l1.split(' ')[-1] + ',' + l2.split(' ')[-1]
            ticks.append(tick)

        with open(os.path.join(root_dir, dir, 'cps.txt'), 'r') as f2:
            cps_hat = f2.readline().split(' ')[0:-1]
            cps_hat = np.array([int(i) for i in cps_hat])
            cps = f2.readline().split(' ')[0:-1]
            cps = np.array([int(i) for i in cps])
            diff = np.abs(cps_hat - cps)


            lower.append(np.percentile(diff, [20]))
            median.append(np.percentile(diff, [50]))
            upper.append(np.percentile(diff, [80]))
            
            diffs.append(diff)
            means.append(np.sum(diff) / len(diff))
            stds.append(np.std(diff))

sorted_labels = sorted(zip(means, ticks), key=lambda pair:pair[0])
means = [x for x,_ in sorted_labels]
ticks = [x for _,x in sorted_labels]

# plot the average of |eta - eta_hat| across test samples
plt.errorbar(ticks, means, yerr=stds, fmt='o')
plt.xticks(ticks, ticks, fontsize=7)
plt.xlabel('beta1, beta2 (style KL, content KL)')
plt.ylabel('mean |eta-eta_hat|')
plt.title(title)
plt.show()
plt.close()

'''
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