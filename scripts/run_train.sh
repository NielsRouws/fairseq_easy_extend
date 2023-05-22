#! /bin/bash

python train.py --config-dir "fairseq_easy_extend/models/nat/" --config-name "cmlm_config.yaml" \
task.data=$(pwd)/data/iwslt14.tokenized.de-en \
checkpoint.restore_file=models/checkpoint_best.pt \
checkpoint.reset_optimizer=True
