#!/bin/sh

python BertCRFVariantDataloaderTest.py \
    --task_name 4CWS_CWS \
    --model_type sequencelabeling \
    --data_dir ../data/CWS/BMES/MSR/ \
    --bert_model_dir ../models/bert-base-chinese/ \
    --vocab_file ../models/bert-base-chinese/vocab.txt \
    --do_lower_case True \
    --max_seq_length 128 \
    --init_checkpoint ../models/bert-base-chinese\
    --override_output True \
    --learning_rate 1e-4 \
    --method last_layer \
    --output_dir ./tmp/4CWS/MSR/CRF_BiLSTM_l11 \
    --num_hidden_layers 11 \
    --train_batch_size 128 \
    --visible_device 0 \
    --num_train_epochs 10
