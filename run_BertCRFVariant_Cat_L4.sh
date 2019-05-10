#!/bin/sh

python BertCRFVariantDataloaderTest.py \
    --task_name PKU \ #  MSR \
    --model_type sequencelabeling \
    --data_dir ../data/CWS/BMES/ \ # need add MSR/
    --bert_model_dir ../models/bert-base-chinese/ \
    --vocab_file ../models/bert-base-chinese/vocab.txt \
    --do_lower_case True \
    --max_seq_length 128 \
    --init_checkpoint ../models/bert-base-chinese\
    --override_output True \
    --learning_rate 1e-4 \
    --method cat_last4 \
    --output_dir ./tmp/4CWS/ \ # need add MSR/CRF/cat_last4
    --num_hidden_layers 12 \
    --train_batch_size 128 \
    --visible_device 3 \
    --num_train_epochs 20

