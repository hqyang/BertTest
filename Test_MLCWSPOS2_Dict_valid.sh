#!/bin/sh
for i in 12,5
do
    IFS=",";
    set -- $i;
    echo $1, $2;

    python BertMLCWSPOS_With_Dict_DataloaderTest.py \
        --task_name ontonotes_cws_pos2.0 \
        --model_type sequencelabeling \
        --data_dir ../data/ontonotes5/4nerpos_update/valid/feat_with_dict/ \
        --output_dir ./tmp/ontonotes/CWSPOS2/cased2/valid/feat_with_dict/ \
        --fclassifier Softmax \
        --bert_model_dir ../models/multi_cased_L-12_H-768_A-12/ \
        --vocab_file ./src/BERT/models/multi_cased_L-12_H-768_A-12/vocab.txt \
        --do_lower_case True \
        --max_seq_length 128 \
        --init_checkpoint ../models/multi_cased_L-12_H-768_A-12/ \
        --override_output True \
        --learning_rate 2e-5 \
        --method fine_tune \
        --num_hidden_layers $1 \
        --train_batch_size $2 \
        --visible_device 3 \
        --num_train_epochs 20
done
