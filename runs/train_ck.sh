#!/bin/bash

# parameters
DATABACK='~/.datasets/coco'
DATA='~/.datasets'
NAMEDATASET='ck'
NAMEMETHOD='attnet' #attnet, attstnnet, attgmmnet, attgmmstnnet, classnet
ARCH='ferattention' #ferattention, ferattentiongmm, ferattentionstn, preactresnet18
PROJECT='../out/'$NAMEMETHOD
EPOCHS=5
TRAINITERATION=1000
TESTITERATION=100
BATCHSIZE=128 #32, 64, 128, 160, 200, 240
LEARNING_RATE=0.0001
MOMENTUM=0.5
PRINT_FREQ=100
WORKERS=4
RESUME='model_best.pth.tar' #chk000000, model_best
GPU=0
LOSS='attloss'
OPT='adam'
SCHEDULER='fixed'
NUMCLASS=8 #6, 7, 8
NUMCHANNELS=3
DIM=32
SNAPSHOT=10
IMAGESIZE=64
KFOLD=5
NACTOR=10
BACKBONE='preactresnet' #preactresnet, resnet, cvgg
NUM_FILTERS=32
BREAL='synthetic' #real, synthetic
POOLSIZE=2 #2,4,8,16
EXP_NAME='feratt_'$NAMEMETHOD'_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_'$BREAL'_filter'$NUM_FILTERS'_pool_size'$POOLSIZE'_dim'$DIM'_bb'$BACKBONE'_fold'$KFOLD'_000'

rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log
rm -rf $PROJECT/$EXP_NAME/
mkdir -p $PROJECT
mkdir -p $PROJECT/$EXP_NAME



CUDA_VISIBLE_DEVICES=0 python ../train.py \
$DATA \
--name-dataset=$NAMEDATASET \
--databack=$DATABACK \
--trainiteration=$TRAINITERATION \
--testiteration=$TESTITERATION \
--project=$PROJECT \
--name=$EXP_NAME \
--epochs=$EPOCHS \
--kfold=$KFOLD \
--nactor=$NACTOR \
--batch-size=$BATCHSIZE \
--learning-rate=$LEARNING_RATE \
--momentum=$MOMENTUM \
--image-size=$IMAGESIZE \
--channels=$NUMCHANNELS \
--dim=$DIM \
--num-classes=$NUMCLASS \
--print-freq=$PRINT_FREQ \
--snapshot=$SNAPSHOT \
--workers=$WORKERS \
--gpu=$GPU \
--loss=$LOSS \
--opt=$OPT \
--scheduler=$SCHEDULER \
--name-method=$NAMEMETHOD \
--arch=$ARCH \
--finetuning \
--breal=$BREAL \
--num_filters=$NUM_FILTERS \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \

#--parallel \
echo $EXP_NAME
