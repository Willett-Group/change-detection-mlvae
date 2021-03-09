import os
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, Index
import json

def build_dataframe(dictionary):
    result = []
    for b1 in beta1:
        row = []
        for b2 in beta2:
            if b1 == b2:
                print(result)
                row.append(dictionary[b1+';'+b2])
        result.append(row)
    return result

config = {
    'experiment_setting': 'cifar10',
    'dataset_dir': 'n=2800T=10',
    'model': 'linearmlvae_50', # 'dfcvae', 'linearvae', 'convvae', 'resnetvae'
}

#root_dir = 'experiments/' + config['experiment_setting']
#title = config['experiment_setting']

#root_dir = 'experiments/' + config['experiment_setting'] + '/' + config['model']
#title = config['experiment_setting'] + ' ' + config['dataset_dir'] + ' ' + config['model']

beta1 = set()
beta2 = set()
diffs, means, stds, lower, median, upper = {},{},{},{},{},{}

'''
for dir in os.listdir(root_dir):
    if dir.isdigit():
        with open(path.join(root_dir, dir, 'args.txt'), 'r') as f1:
            args = json.load(f1)
            print(args)
            b1 = str(args['beta1'])
            b2 = str(args['beta2'])
            beta1.add(b1)
            beta2.add(b2)
            key = b1 + ';' + b2

        with open(path.join(root_dir, dir, 'cps.txt'), 'r') as f2:
            cps_hat = f2.readline().split(' ')[0:-1]
            cps_hat = np.array([int(i) for i in cps_hat])
            cps = f2.readline().split(' ')[0:-1]
            cps = np.array([int(i) for i in cps])
            diff = np.abs(cps_hat - cps)

            diffs[key] = diff
            means[key] = np.sum(diff) / len(diff)
            stds[key] = np.std(diff)
            lower[key] = np.percentile(diff, [25])
            median[key] = np.percentile(diff, [50])
            upper[key] = np.percentile(diff, [80])
'''
diffs = {'0.001;0.001': 5.2, '0.00001;0.00001': 4.466, '1.0;1.0': 11.86,
         '10;10':11.9, '0.0000001;0.0000001':5.66}
beta1 = ('0.0000001', '0.00001', '0.001', '1.0', '10')
beta2 = ('0.0000001', '0.00001', '0.001', '1.0', '10')
beta1 = sorted(beta1)
beta2 = sorted(beta2)
diffs_data = build_dataframe(diffs)
means_data = build_dataframe(means)
stds_data = build_dataframe(stds)
lower_data = build_dataframe(lower)
median_data = build_dataframe(median)
upper_data = build_dataframe(upper)

data_to_plot = [means_data, stds_data]
type = ['mean', 'std']
for i in range(len(data_to_plot)):
    data = data_to_plot[i]
    idx = Index(beta1)
    df = DataFrame(np.asarray(data), index=idx, columns=beta2)
    vals = np.around(df.values, 2)
    norm = plt.Normalize(vals.min()-1, vals.max()+1)
    colors = plt.cm.hot(norm(vals))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])

    plt.table(cellText=vals, rowLabels=df.index, colLabels=df.columns,
              colWidths = [0.15]*vals.shape[1], loc='center', cellColours=colors)
    plt.xlabel("beta2")
    plt.ylabel("beta1")
    # plt.title(config['experiment_setting']+'; '+config['dataset_dir']+'; '+config['model']+'\n'+type[i])
    plt.title(config['experiment_setting']+'; '+config['model']+'\n'+type[i])
    plt.show()