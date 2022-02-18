python test.py \
--dataset cifar100 \
--n_max 200 \
--t_max 50 \
--seed 0 \
--model_path TRAIN-20220213-214531 \
--method siamese_p \
--loss_variant 1

python test.py \
--dataset cifar10 \
--n_max 500 \
--t_max 20 \
--seed 0 \
--model_path TRAIN-20220213-200526 \
--method siamese_p \
--loss_variant 0

python test.py \
--dataset cifar100 \
--n_max 100 \
--t_max 100 \
--seed 0 \
--model_path TRAIN-20220213-214531 \
--method siamese_p \
--loss_variant 1

python test.py \
--dataset cifar100 \
--n_max 40 \
--t_max 250 \
--seed 0 \
--model_path TRAIN-20220213-214531 \
--method siamese_p \
--loss_variant 1

python test.py \
--dataset cifar10 \
--n_max 20 \
--t_max 500 \
--seed 0 \
--model_path TRAIN-20220213-200526 \
--method siamese_p \
--loss_variant 0