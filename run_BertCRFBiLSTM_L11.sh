#!/bin/sh

python BertCRFCWSDataloaderBiLSTMTest.py \
    --task_name 4CWS_CWS \
    --model_type sequencelabeling \
    --data_dir ../data/CWS/BMES/MSR/ \
    --bert_model_dir ../models/bert-base-chinese/ \
    --vocab_file ../models/bert-base-chinese/vocab.txt \
    --output_dir ./tmp/4CWS/MSR/CRF_BiLSTM_l11 \
    --do_lower_case True \
    --max_seq_length 128 \
    --num_hidden_layers 11 \
    --init_checkpoint ../models/bert-base-chinese\
    --train_batch_size 32 \
    --override_output True \
    --tensorboardWriter False \
    --visible_device 0 \
    --num_train_epochs 10 \
    --bfinetune False \
    --learning_rate 1e-4

#     --bert_config_file ../models/bert-base-chinese/bert_config.json,