#!/bin/bash

PATHDATASET='~/.datasets/'
NAMEDATASET='ck' #bu3dfe, ferblack, ck, affectnetdark, affectnet, ferp
PROJECT='../out/attnet'
FILENAME='result.txt'
PATHMODEL='models'
NAMEMODEL='model_best.pth.tar' #'model_best.pth.tar' #'chk000565.pth.tar'

PROJECTNAME='feratt_attnet_ferattention_attloss_adam_ck_dim32_bbpreactresnet_fold0_000'
MODEL=$PROJECT/$PROJECTNAME/$PATHMODEL/$NAMEMODEL

python ../eval.py \
--project=$PROJECT \
--projectname=$PROJECTNAME \
--pathdataset=$PATHDATASET \
--namedataset=$NAMEDATASET \
--filename=$FILENAME \
--model=$MODEL \
