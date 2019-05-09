#!/bin/sh

python BertCRFCWSDataloaderBiLSTMTest.py \
    --task_name 4CWS_CWS \
    --model_type sequencelabeling \
    --data_dir ../data/CWS/BMES/MSR/ \
    --bert_model_dir ../models/bert-base-chinese/ \
    --vocab_file ../models/bert-base-chinese/vocab.txt \
    --output_dir ./tmp/4CWS/MSR/CRF_BiLSTM_SumL4 \
    --do_lower_case True \
    --max_seq_length 128 \
    --num_hidden_layers 12 \
    --init_checkpoint ../models/bert-base-chinese\
    --train_batch_size 128 \
    --override_output True \
    --tensorboardWriter False \
    --visible_device 1 \
    --num_train_epochs 10 \
    --bfinetune False \
    --method sum_last4 \
    --learning_rate 1e-4

#     --bert_config_file ../models/bert-base-chinese/bert_config.json,
