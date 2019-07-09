#!/bin/sh
for nhl in 1 3 6 12
do
    python BertMLCWSPOSDataloaderTest.py \
        --task_name ontonotes_cws_pos2.0 \
        --model_type sequencelabeling \
        --data_dir ../data/ontonotes5/4nerpos_update \
        --output_dir ./tmp/ontonotes/CWSPOS2/ \
        --fclassifier Softmax \
        --bert_model_dir ../models/multi_cased_L-12_H-768_A-12/ \
        --vocab_file ./src/BERT/models/multi_cased_L-12_H-768_A-12/vocab.txt \
        --do_lower_case True \
        --max_seq_length 128 \
        --init_checkpoint ../models/multi_cased_L-12_H-768_A-12/ \
        --override_output True \
        --learning_rate 2e-5 \
        --method fine_tune \
        --num_hidden_layers $nhl \
        --train_batch_size 32 \
        --visible_device 3 \
        --num_train_epochs 10
done
