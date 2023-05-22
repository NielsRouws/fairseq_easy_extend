#! /bin/bash

python decode.py ./data/iwslt14.tokenized.de-en --source-lang de --target-lang en \
--path models/checkpoint_best.pt \
--task translation_lev \
--iter-decode-max-iter 9 \
--gen-subset test \
--print-step \
--remove-bpe \
--tokenizer moses \
--scoring bleu
