#!/bin/bash

# parameters
DATABACK=$HOME'/.datasets/coco'
DATA=$HOME'/.datasets'
NAMEDATASET='ck'
PROJECT='../out/multiple_attnet'
EPOCHS=60
TRAINITERATION=1000
TESTITERATION=100
BATCHSIZE=200 #32, 64, 128, 160, 200, 240
LEARNING_RATE=0.0001
MOMENTUM=0.5
PRINT_FREQ=100
WORKERS=4
RESUME='model_best.pth.tar' #chk000000, model_best
GPU=1
NAMEMETHOD='attnet' #attnet, attstnnet, attgmmnet, attgmmstnnet
ARCH='ferattention' #ferattention, ferattentiongmm, ferattentionstn
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
BREAL='real' #real, synthetic
EXP_NAME='beta_feratt_'$NAMEMETHOD'_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_'$BREAL'_dim'$DIM'_bb'$BACKBONE'_fold'$KFOLD'_000'

rm -rf $PROJECT/$EXP_NAME/
mkdir -p $PROJECT
mkdir -p $PROJECT/$EXP_NAME

for iteration in 1 2 3 4 5
do
  for run in 1 2 3 4 5
  do

  if [ $iteration = 1 ]
  then
  alpha=2
  beta=4
  fi
  if [ $iteration = 2 ]
  then
  alpha=4
  beta=2
  fi
  if [ $iteration = 3 ]
  then
  alpha=1
  beta=1
  fi
  if [ $iteration = 4 ]
  then
  alpha=.7
  beta=.7
  fi
  if [ $iteration = 5 ]
  then
  alpha=4
  beta=4
  fi
echo "The following iteration had "$alpha' and '$beta' as its parameters.' | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log
echo "It is run "$run" of 5."
CUDA_VISIBLE_DEVICES=1 python ../train.py \
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
--resume=$RESUME \
--gpu=$GPU \
--loss=$LOSS \
--opt=$OPT \
--scheduler=$SCHEDULER \
--name-method=$NAMEMETHOD \
--arch=$ARCH \
--finetuning \
--breal=$BREAL \
--alpha=$alpha \
--beta=$beta \
2>&1 | tee -a $PROJECT/$EXP_NAME/$EXP_NAME.log \

#--parallel \
echo $EXP_NAME

done
done
