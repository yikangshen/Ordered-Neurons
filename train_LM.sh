#!/bin/bash
source ./hyperparameters.txt

python -u $lm_dir/main.py \
    --lm_data $lm_data_dir \
    --cuda \
    --epochs $epochs \
    --model $model \
    --nlayers $nlayers \
    --nhid $num_hid \
    --emsize $emsize \
    --save $model_dir/lstm_lm.pt \
    --save_lm_data $model_dir/lstm_lm.bin \
    --log-interval $log_freq \
    --batch $batch_size \
    --dropout $dropout \
    --dropouth $dropout \
    --dropouti $dropout \
    --lr $lr
