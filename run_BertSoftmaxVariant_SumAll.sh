#!/bin/sh

python BertSoftmaxVariantDataloaderTest.py \
    --task_name PKU \
    --model_type sequencelabeling \
    --data_dir ../data/CWS/BMES/ \
    --output_dir ./tmp/4CWS/ \
    --bert_model_dir ../models/bert-base-chinese/ \
    --vocab_file ../models/bert-base-chinese/vocab.txt \
    --do_lower_case True \
    --max_seq_length 128 \
    --init_checkpoint ../models/bert-base-chinese\
    --override_output True \
    --learning_rate 2e-5 \
    --method sum_all \
    --num_hidden_layers 12 \
    --train_batch_size 128 \
    --visible_device 2 \
    --num_train_epochs 30
