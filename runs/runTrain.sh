#!/bin/bash

# parameters
DATABACK='~/.datasets/coco'
DATA='~/.datasets'
NAMEDATASET='ckdark' #affectnetdark, ckdark, bu3dfedark, jaffedark
PROJECT='../out/attnet'
EPOCHS=500
TRAINITERATION=288000
TESTITERATION=2880
BATCHSIZE=32 #32, 64, 128, 192, 256
LEARNING_RATE=0.0001
MOMENTUM=0.5
PRINT_FREQ=100
WORKERS=4
RESUME='chk000000.pth.tar' #chk000000, model_best
GPU=1
NAMEMETHOD='attnet' # attnet, attstnnet, attgmmnet, attgmmstnnet
ARCH='ferattention' # ferattention, ferattentionstn, ferattentiongmm, ferattentiongmmstn
LOSS='attloss'
OPT='adam'
SCHEDULER='fixed' #fixed, step, plateau
NUMCLASS=8 #6, 7, 8
NUMCHANNELS=3
DIM=32
SNAPSHOT=10
IMAGESIZE=64
KFOLD=0
NACTOR=10
BACKBONE='preactresnet' #preactresnet, resnet, cvgg

EXP_NAME='feratt_'$NAMEMETHOD'_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_dim'$DIM'_bb'$BACKBONE'_fold'$KFOLD'_000'

rm -rf $PROJECT/$EXP_NAME/$EXP_NAME.log # This line seems kind of pointless, since we do this as part of the next
rm -rf $PROJECT/$EXP_NAME/
mkdir -p $PROJECT
mkdir -p $PROJECT/$EXP_NAME

#0,1,2,3
CUDA_VISIBLE_DEVICES=1


for number in 2 4 8 16 32
  do
  python ../train.py \
  $DATA \
  --databack=$DATABACK \
  --name-dataset=$NAMEDATASET \
  --project=$PROJECT \
  --name=$EXP_NAME \
  --epochs=$EPOCHS \
  --trainiteration=$TRAINITERATION \
  --testiteration=$TESTITERATION \
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
  --resume=$RESUME \
  --gpu=$GPU \
  --loss=$LOSS \
  --opt=$OPT \
  --scheduler=$SCHEDULER \
  --name-method=$NAMEMETHOD \
  --arch=$ARCH \
  --finetuning \
  --aveNum=$number \
  | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log
  done



#--parallel \