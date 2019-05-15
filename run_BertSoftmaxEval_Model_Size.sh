#!/bin/sh

for nhl in 1 3 6 12
do
    python eval.py \
        --task_name PKU \
        --model_type sequencelabeling \
        --data_dir ../data/CWS/BMES/ \
        --output_dir ./tmp/4CWS/ModelSize/ \
        --fclassifier Softmax \
        --bert_model_dir ../models/bert-base-chinese/ \
        --vocab_file ../models/bert-base-chinese/vocab.txt \
        --do_eval_df True \
        --do_lower_case True \
        --max_seq_length 128 \
        --init_checkpoint ../models/bert-base-chinese \
        --override_output True \
        --learning_rate 2e-5 \
        --method fine_tune \
        --num_hidden_layers $nhl \
        --train_batch_size 32 \
        --visible_device 2 
done
# --task_name PKU \ #  MSR \
# --data_dir ../data/CWS/BMES/ # need add MSR/
# --output_dir ./tmp/4CWS/ \  #need add MSR/CRF/fine_tune
# small dataset lr = 2e-5
