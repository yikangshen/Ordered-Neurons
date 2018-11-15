#!/bin/bash
source ./hyperparameters.txt

python $workdir/src/LM_eval.py \
--model_type rnn \
--model $model_dir/lstm_lm.pt \
--lm_data $model_dir/lstm_lm.bin
