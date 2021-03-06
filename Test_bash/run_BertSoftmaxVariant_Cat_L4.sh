#!/bin/sh

python BertVariantDataloaderTest.py \
    --task_name PKU \
    --model_type sequencelabeling \
    --data_dir ../data/CWS/BMES/ \
    --output_dir ./tmp/4CWS/ \
    --fclassifier Softmax \
    --bert_model_dir ../models/bert-base-chinese/ \
    --vocab_file ../models/bert-base-chinese/vocab.txt \
    --do_lower_case True \
    --max_seq_length 128 \
    --init_checkpoint ../models/bert-base-chinese\
    --override_output True \
    --learning_rate 2e-5 \
    --method cat_last4 \
    --num_hidden_layers 12 \
    --train_batch_size 128 \
    --visible_device 0 \
    --num_train_epochs 30

