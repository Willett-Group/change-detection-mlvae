# change-detection-mlvae

Training files:
train_mnist.py
train_celeba.py
train_cifar10.py

Testing files: (approach 2)
test_mnist.py
test_celeba.py
test_cifar10.py

Every training and testing file contains a "config" dictionary that stores the information about the experiment
and hyperparemeters of the experiment.
    experiment_name: name of the this experiment, will create a directory with this name in "experiments/"
    experiment_type: repetitive or nonrepetitive. The former means all images in a group are the same.
    model: convvae, linearvae, dfcvae, or resnetvae.
          convvae uses simple naive convolutional layers
          linearvae uses simple naive linear layers
          dfcvae is deep-feature consistent vae
          resnetvae uses a resnet as a first step in the encoding state
    n: The value of N (number of time series samples)
    T: The value of T
    
    start_epoch: start epoch
    end_epoch: end epoch
    b_size: batch size
    initial_lr: initial learning rate
    beta: coefficient of KL-divergence term

    log_file: name of log file
    load_saved: If true, load saved models from log file. Should change start_epoch accordingly.
