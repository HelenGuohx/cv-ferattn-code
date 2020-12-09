#!/bin/bash

# parameters
DATABACK='~/.datasets/coco'
DATA='~/.datasets'
NAMEDATASET='ferp'
NAMEMETHOD='attnet' #attnet, attstnnet, attgmmnet, attgmmstnnet
ARCH='ferattention' #ferattention, ferattentiongmm, ferattentionstn
PROJECT='../out/attnet'
#TRAINITERATION=10000
#TESTITERATION=1000
#EPOCHS=10
#BATCHSIZE=128 #32, 64, 128, 160, 200, 240
#LEARNING_RATE=0.0001
#MOMENTUM=0.5
#PRINT_FREQ=100
##WORKERS=4
#RESUME='model_best.pth.tar' #chk000000, model_best
#GPU=0
LOSS='attloss'
OPT='adam'
#SCHEDULER='fixed'
#NUMCLASS=8 #6, 7, 8
#NUMCHANNELS=3
DIM=32
#SNAPSHOT=10
#IMAGESIZE=64
KFOLD=5
#NACTOR=10
BACKBONE='preactresnet' #preactresnet, resnet, cvgg
BREAL='synthetic' #real, synthetic

# change the recontruction size(set by num_filters in neural network) defaul is 32
for nf in 64 128
do
  echo "num filters="$nf

  NUM_FILTERS=$nf

  EXP_NAME='feratt_'$NAMEMETHOD'_'$ARCH'_'$LOSS'_'$OPT'_'$NAMEDATASET'_'$BREAL'_filter'$NUM_FILTERS'_dim'$DIM'_bb'$BACKBONE'_fold'$KFOLD'_000'

  if [ $NAMEDATASET = 'ck' ]
  then
    sh train_ck.sh $NUM_FILTERS

    sh eval_ck.sh $EXP_NAME 2>&1 | tee -a $PROJECT/$EXP_NAME/'eval.log'

  elif [ $NAMEDATASET = 'ferp' ]
  then
    sh train_ferp.sh $NUM_FILTERS

    sh eval_ferp.sh $EXP_NAME 2>&1 | tee -a $PROJECT/$EXP_NAME/'eval.log'

  else
    echo "$NAMEDATASET is not defined"
    exit 1
  fi
done

