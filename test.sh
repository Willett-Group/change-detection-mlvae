dataset="cifar100"
path="cifar100/seed_0/TRAIN-20220219-010639"

python test.py \
--dataset $dataset \
--n_max 200 \
--t_max 50 \
--seed 0 \
--model_path $path \
--method siamese_logp \
--loss_variant 1

python test.py \
--dataset $dataset \
--n_max 500 \
--t_max 20 \
--seed 0 \
--model_path $path \
--method siamese_logp \
--loss_variant 1