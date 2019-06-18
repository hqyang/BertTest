#!/bin/sh

python BertCWSPOSDataloaderTest.py \
    --task_name ontonotes \
    --model_type sequencelabeling \
    --data_dir ../data/ontonotes5/4nerpos_data \
    --output_dir ./tmp/ontonotes \
    --fclassifier Softmax \
    --bert_model_dir ../models/bert-base-chinese/ \
    --vocab_file ../models/bert-base-chinese/vocab.txt \
    --do_lower_case True \
    --max_seq_length 128 \
    --init_checkpoint ../models/bert-base-chinese \
    --override_output True \
    --learning_rate 2e-5 \
    --method fine_tune \
    --num_hidden_layers 12 \
    --train_batch_size 32 \
    --visible_device 0