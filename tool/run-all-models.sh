#!/bin/bash


for arch in 'alexnet' 'resnet18' 'shufflenet_v2_x0_5' 'mobilenet_v2' 'squeezenet1_0'; do
    for workers in 3; do
         for num_gpu in 1 2 4; do
             python harness.py --nproc_per_node=$num_gpu -j $workers -b 512  -a $arch --prefix results/dali-gpu/  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /datadrive/mnt2/jaya/datasets/imagenet/
         done
    done
done

for arch in 'resnet50' 'vgg11'; do
    for workers in 3; do
         for num_gpu in 1 2 4; do
             python harness.py --nproc_per_node=$num_gpu -j $workers -b 512  -a $arch --prefix results/dali-cpu/  image_classification/pytorch-imagenet-dali-mp.py --dali_cpu --amp --noeval  --data /datadrive/mnt2/jaya/datasets/imagenet/
         done
    done
done



for arch in 'alexnet' 'resnet18' 'shufflenet_v2_x0_5' 'mobilenet_v2' 'squeezenet1_0'; do
    for workers in 1 2 6; do
         for num_gpu in 1; do
             python harness.py --nproc_per_node=$num_gpu -j $workers -b 512  -a $arch --prefix results/dali-gpu/w${workers}  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /datadrive/mnt2/jaya/datasets/imagenet/
         done
    done
done

for arch in 'resnet50' 'vgg11'; do
    for workers in 1 2 6; do
         for num_gpu in 1; do
             python harness.py --nproc_per_node=$num_gpu -j $workers -b 512  -a $arch --prefix results/dali-cpu/w${workers}/  image_classification/pytorch-imagenet-dali-mp.py --dali_cpu --amp --noeval  --data /datadrive/mnt2/jaya/datasets/imagenet/
         done
    done
done


for arch in 'alexnet' 'resnet18' 'shufflenet_v2_x0_5' 'mobilenet_v2' 'squeezenet1_0'; do
    for workers in 6; do
         for num_gpu in 4; do
             python harness.py --nproc_per_node=$num_gpu -j $workers -b 512  -a $arch --prefix results/dali-gpu/w${workers}  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /datadrive/mnt2/jaya/datasets/imagenet/
         done
    done
done

for arch in 'resnet50' 'vgg11'; do
    for workers in 6; do
         for num_gpu in 4; do
             python harness.py --nproc_per_node=$num_gpu -j $workers -b 512  -a $arch --prefix results/dali-cpu/w${workers}/  image_classification/pytorch-imagenet-dali-mp.py --dali_cpu --amp --noeval  --data /datadrive/mnt2/jaya/datasets/imagenet/
         done
    done
done


for arch in 'alexnet' 'resnet18' 'shufflenet_v2_x0_5' 'mobilenet_v2' 'squeezenet1_0'; do
    for workers in 12; do
         for num_gpu in 2; do
             python harness.py --nproc_per_node=$num_gpu -j $workers -b 512  -a $arch --prefix results/dali-gpu/w${workers}  image_classification/pytorch-imagenet-dali-mp.py --amp --noeval  --data /datadrive/mnt2/jaya/datasets/imagenet/
         done
    done
done

for arch in 'resnet50' 'vgg11'; do
    for workers in 12; do
         for num_gpu in 2; do
             python harness.py --nproc_per_node=$num_gpu -j $workers -b 512  -a $arch --prefix results/dali-cpu/w${workers}/  image_classification/pytorch-imagenet-dali-mp.py --dali_cpu --amp --noeval  --data /datadrive/mnt2/jaya/datasets/imagenet/
         done
    done
done
