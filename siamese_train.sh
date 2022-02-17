# python siamese_train.py \
# --set cifar10 \
# --n_max 1000 \
# --t_max 50 \
# --p_max 50 \
# --epochs 100 \
# --batch_size 128 \
# --report_freq 50 \
# --lr 0.001 \
# --seed 0

python siamese_train.py \
--set cifar10 \
--n_max 5000 \
--t_max 10 \
--p_max 10 \
--epochs 100 \
--batch_size 128 \
--report_freq 50 \
--lr 0.001 \
--seed 0

python siamese_train.py \
--set cifar10 \
--n_max 500 \
--t_max 100 \
--p_max 100 \
--epochs 100 \
--batch_size 128 \
--report_freq 50 \
--lr 0.001 \
--seed 0

python siamese_train.py \
--set cifar10 \
--n_max 100 \
--t_max 500 \
--p_max 500 \
--epochs 100 \
--batch_size 128 \
--report_freq 50 \
--lr 0.001 \
--seed 0

python siamese_train.py \
--set cifar10 \
--n_max 50 \
--t_max 1000 \
--p_max 1000 \
--epochs 100 \
--batch_size 128 \
--report_freq 50 \
--lr 0.001 \
--seed 0
