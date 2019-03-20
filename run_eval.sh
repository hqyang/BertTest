#!/bin/sh

for nhl in 3 6
do
    for nte in 15
    do
        for tbs in 64
        do 
            echo "train_batch_size $tbs, num_train_epochs $nte, num_hidden_layers $nhl"
            python main.py \
             --task_name ontonotes_CWS \
             --model_type sequencelabeling \
             --data_dir ../data/ontonotes5/ \
             --bert_model_dir ../models/bert-base-chinese/ \
             --vocab_file ../models/bert-base-chinese/vocab.txt \
             --output_dir ./tmp/ontonotes \
             --do_train False \
             --init_checkpoint ./tmp_2019_3_20/ontonotes/ \
             --do_eval True \
             --do_lower_case True \
             --train_batch_size $tbs \
             --override_output True \
             --tensorboardWriter False \
             --visible_device 0 \
             --num_train_epochs $nte \
             --max_seq_length 128 \
             --num_hidden_layers $nhl
         done
    done
done
#eval_BERT_rs.py \ 
