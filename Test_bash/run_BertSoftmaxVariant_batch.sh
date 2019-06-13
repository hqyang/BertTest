#!/bin/sh

# 0
./run_BertSoftmaxVariant_AllFinetune.sh
./run_BertSoftmaxVariant_L1.sh

# 1
./run_BertSoftmaxVariant_L11.sh
./run_BertSoftmaxVariant_L12.sh

# 2
./run_BertSoftmaxVariant_SumL4.sh
./run_BertSoftmaxVariant_SumAll.sh

# 3
./run_BertSoftmaxVariant_Cat_L4.sh
