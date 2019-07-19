#!/bin/sh
for i in 1,256 3,128 6,64 12,32
do
    IFS=",";
    set -- $i;
    echo $1, $2;

    python BertCWSPOSDataloaderTest.py \
        --task_name ontonotes_cws_pos \
        --model_type sequencelabeling \
        --data_dir ../data/ontonotes5/4nerpos_data \
        --output_dir ./tmp/ontonotes \
        --fclassifier Softmax \
        --bert_model_dir ../models/multi_cased_L-12_H-768_A-12/ \
        --vocab_file ../models/multi_cased_L-12_H-768_A-12/vocab.txt \
        --do_lower_case False \
        --max_seq_length 128 \
        --init_checkpoint ../models/multi_cased_L-12_H-768_A-12 \
        --override_output True \
        --learning_rate 2e-5 \
        --method fine_tune \
        --num_hidden_layers $1 \
        --train_batch_size $2 \
        --visible_device 0 \
        --num_train_epochs 20
done

#         --init_checkpoint ../models/bert-base-chinese \
#         --vocab_file ../models/bert-base-chinese/vocab.txt \
#         --bert_model_dir ../models/bert-base-chinese/ \
