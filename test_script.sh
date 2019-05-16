# CRF
mkdir CRF
chmod 777 CRF

mkdir CRF_BiLSTM_CatL4
chmod 777 CRF_BiLSTM_CatL4

mkdir CRF_BiLSTM_SumAll
chmod 777 CRF_BiLSTM_SumAll

mkdir CRF_BiLSTM_SumL4
chmod 777 CRF_BiLSTM_SumL4

mkdir CRF_BiLSTM_l1
chmod 777 CRF_BiLSTM_l1

mkdir CRF_BiLSTM_l11
chmod 777 CRF_BiLSTM_l11

mkdir CRF_BiLSTM_l12
chmod 777 CRF_BiLSTM_l12

# Softmax
mkdir Softmax
chmod 777 Softmax

mkdir Softmax_BiLSTM_CatL4
chmod 777 Softmax_BiLSTM_CatL4

mkdir Softmax_BiLSTM_SumAll
chmod 777 Softmax_BiLSTM_SumAll

mkdir Softmax_BiLSTM_SumL4
chmod 777 Softmax_BiLSTM_SumL4

mkdir Softmax_BiLSTM_l1
chmod 777 Softmax_BiLSTM_l1

mkdir Softmax_BiLSTM_l11
chmod 777 Softmax_BiLSTM_l11

mkdir Softmax_BiLSTM_l12
chmod 777 Softmax_BiLSTM_l12

# add folders in Softmax
mkdir Softmax/finetune
chmod 777 Softmax/finetune

mkdir Softmax/BiLSTM_CatL4
chmod 777 Softmax/BiLSTM_CatL4

mkdir Softmax/BiLSTM_SumAll
chmod 777 Softmax/BiLSTM_SumAll

mkdir Softmax/BiLSTM_SumL4
chmod 777 Softmax/BiLSTM_SumL4

mkdir Softmax/BiLSTM_l1
chmod 777 Softmax/BiLSTM_l1

mkdir Softmax/BiLSTM_l11
chmod 777 Softmax/BiLSTM_l11

mkdir Softmax/BiLSTM_l12
chmod 777 Softmax/BiLSTM_l12

# add folders in CRF
mkdir CRF/finetune
chmod 777 CRF/finetune

mkdir CRF/BiLSTM_CatL4
chmod 777 CRF/BiLSTM_CatL4

mkdir CRF/BiLSTM_SumAll
chmod 777 CRF/BiLSTM_SumAll

mkdir CRF/BiLSTM_SumL4
chmod 777 CRF/BiLSTM_SumL4

mkdir CRF/BiLSTM_l1
chmod 777 CRF/BiLSTM_l1

mkdir CRF/BiLSTM_l11
chmod 777 CRF/BiLSTM_l11

mkdir CRF/BiLSTM_l12
chmod 777 CRF/BiLSTM_l12

git add run_BertSoftmaxVariant_L1.sh run_BertSoftmaxVariant_L11.sh run_BertSoftmaxVariant_L12.sh run_BertSoftmaxVariant_SumL4.sh run_BertSoftmaxVariant_SumAll.sh run_BertSoftmaxVariant_Cat_L4.sh run_BertSoftmaxVariant_AllFinetune.sh run_BertSoftmaxVariant_batch.sh
git add run_BertCRFVariant_L1.sh run_BertCRFVariant_L11.sh run_BertCRFVariant_L12.sh run_BertCRFVariant_SumL4.sh run_BertCRFVariant_SumAll.sh run_BertCRFVariant_Cat_L4.sh run_BertCRFVariant_AllFinetune.sh

run_BertCRFVariant_batch.sh
