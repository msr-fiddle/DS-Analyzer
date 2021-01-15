#!/bin/bash

if [ "$#" -ne 5 ]; then
  echo "Usage : ./run-all-workers <data-dir> <out-dir> <num-gpu> <num-cpu> <classes>"
  exit 1
fi

DATA_DIR=$1
OUT_DIR=$2
GPU=$3
CPU=$4
CLASSES=$5

CPU_PER_GPU=$((CPU / GPU))

#for arch in 'resnet18' 'squeezenet1_0' 'vgg11'; do
for arch in 'alexnet' 'resnet18' 'shufflenet_v2_x0_5' 'mobilenet_v2' 'squeezenet1_0' 'resnet50' 'vgg11'; do
#for arch in 'alexnet' 'shufflenet_v2_x0_5' 'mobilenet_v2' 'resnet50' ; do
    for workers in $CPU_PER_GPU; do
         for num_gpu in $GPU; do
             python harness.py --nproc_per_node=$num_gpu -j $workers -b 512  -a $arch --prefix $OUT_DIR/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval --classes $CLASSES --data $DATA_DIR
         done
    done
done



for arch in 'alexnet' 'resnet18' 'shufflenet_v2_x0_5' 'mobilenet_v2' 'squeezenet1_0' 'resnet50' 'vgg11'; do
#for arch in 'squeezenet1_0' 'vgg11'; do
#for arch in 'alexnet' 'shufflenet_v2_x0_5' 'resnet18' 'mobilenet_v2' 'resnet50' ; do
    for workers in $CPU_PER_GPU; do
         for num_gpu in $GPU; do
             python harness.py --nproc_per_node=$num_gpu -j $workers -b 512  -a $arch --prefix $OUT_DIR/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --dali_cpu --classes $CLASSES --data $DATA_DIR
         done
    done
done
