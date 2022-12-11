#!/bin/bash

T=`date +%m%d%H%M%S`

mkdir exp
mkdir exp/$T
mkdir exp/$T/code
cp -r datasets exp/$T/code/datasets
cp -r models exp/$T/code/models
cp ./*.py exp/$T/code/
cp run.sh exp/$T/code

mkdir exp/$T/train.log

datapath=/data/home/elin24/Hercules/coey/gsc147

python main.py --data-path $datapath --batch-size 16 --accumulation-steps 1 --tag $T 2>&1 | tee exp/$T/train.log/running.log
# python main.py --data-path /scratch/wlin38/coey/gsc147/ --batch-size 1 --eval --resume /home/wlin38/scratch/coey/cutpaste/pscc/exp/0710115827/output/ckpt_epoch_best.pth --tag $T 2>&1 | tee exp/$T/train.log/running.log
