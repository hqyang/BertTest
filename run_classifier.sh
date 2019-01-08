#!/bin/bash
export GLUE_DIR=../datasets/glue_data
export BERT_BASE_DIR=./models
python ./torch_codes/run_classifier_torch.py \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --visible_device 1   \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir tmp/mrpc_output/
