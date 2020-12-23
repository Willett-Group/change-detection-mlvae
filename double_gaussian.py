import numpy as np
import pandas as pd
import pickle
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dimension', type=int, default=1, help='dimension of the normal data')
parser.add_argument('--save_csv', type=bool, default=False, help='whether to save the data in csv format')
FLAGS = parser.parse_args()


# fixed variance, mean generated differently
# mean1 is tuple or list
# mean2 is int or list
def generate(T, var, theta, n, m1=(0, 1), m2=1):
    llimit = int(T * 0.1)
    rlimit = int(T * 0.9)
    etas = np.array(range(llimit, rlimit + 1))
    samples, labels = [], []

    for i in range(n):
        eta = np.random.choice(etas, 1)[0]
        if isinstance(m1, tuple):
            mean1 = np.random.uniform(m1[0], m1[1])
        elif isinstance(m1, list):
            mean1 = np.random.choice(m1, 1)[0]
        else:
            raise Exception('invalid mean1 type')

        if theta == -1:
            theta = np.random.uniform(0, 1)

        if isinstance(m2, int):
            mean2 = mean1 + theta * m2 * np.random.choice([1, -1], 1)[0]
        elif isinstance(m2, list):
            mean2 = np.random.choice(m2, 1)[0]
        else:
            raise Exception('invalid mean2 type')

        p1 = np.array(
            [np.random.normal(m, s, eta) for m, s in zip(mean1 * np.ones(784), var * np.ones(784))]
        )
        p2 = np.array(
            [np.random.normal(m, s, T - eta) for m, s in zip(mean2 * np.ones(784), var * np.ones(784))]
        )
        sample = np.concatenate((p1, p2), axis=1)

        samples.append(sample)
        labels.append(eta)

    return samples, labels


def generatewrapper(T, var):
    thetas = [1, 0.75, 0.5, 0.25, -1]
    ns = [1000]
    mean1s = [(0, 1)]
    mean2s = [1]
    datasets_dict = {}

    for i in range(len(mean1s)):
        mean1 = mean1s[i]
        mean2 = mean2s[i]

        for theta in thetas:
            print(theta)
            # generate test data
            x_test, y_test = generate(T, var, theta, 100, mean1, mean2)
            for n in ns:
                name = 'T={}_var={}_theta={}_n={}_m1={}_m2={}'.format(T, var, theta, n, mean1, mean2)
                x_train, y_train = generate(T, var, theta, n, mean1, mean2)
                datasets_dict[name] = (np.array(x_train), np.array(y_train),
                                       np.array(x_test), np.array(y_test))
            # if mean2 is chosen from a discrete set of values, no need to use theta
            if isinstance(mean2, list):
                break

    return datasets_dict


if __name__ == "__main__":
    root_dir = os.getcwd()

    dirs = [os.path.join(root_dir, 'data/')]
    for dir in dirs:
        if not os.path.isdir(dir):
            os.mkdir(dir)

    datasets_dict = generatewrapper(50, 0.2)
    for dsname in datasets_dict:
        # save training and test data to pickle
        with open(os.path.join(dirs[0], dsname), 'wb') as f:
            pickle.dump(datasets_dict[dsname], f)
        # save only test data to csv files for R
        if FLAGS.save_csv:
            pd.DataFrame(datasets_dict[dsname][2]).T.to_csv(dir2 + dsname + '_x_test.csv')
            pd.DataFrame(datasets_dict[dsname][3]).T.to_csv(dir2 + dsname + '_y_test.csv')
