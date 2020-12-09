#!/bin/bash

PATHDATASET='~/.datasets/'
NAMEDATASET='ck' #bu3dfe, ferblack, ck, affectnetdark, affectnet, ferp
NAMEMETHOD='attnet' #attnet, attstnnet, attgmmnet, attgmmstnnet
PROJECT="../out/$NAMEMETHOD"
FILENAME='result.txt'
PATHMODEL='models'
NAMEMODEL='model_best.pth.tar' #'model_best.pth.tar' #'chk000565.pth.tar'
BREAL='real'
PROJECTNAME=$1
MODEL=$PROJECT/$PROJECTNAME/$PATHMODEL/$NAMEMODEL

python ../eval.py \
--project=$PROJECT \
--projectname=$PROJECTNAME \
--pathdataset=$PATHDATASET \
--namedataset=$NAMEDATASET \
--filename=$FILENAME \
--model=$MODEL \
--breal=$BREAL \
--name-method=$NAMEMETHOD \
