# Analyzing and Mitigating Data Stalls in DNN Training

This repository contains the source code implementation of the VLDB'21 paper "Analyzing and Mitigating Data Stalls in DNN Training". This work was done as part of Microsoft Research's [Project Fiddle](https://www.microsoft.com/en-us/research/project/fiddle/). This source code is available under the [MIT License](LICENSE.txt).

We present the first comprehensive analysis of how the data pipeline affects the training of the widely used Deep Neural Networks (DNNs). We find that in
many cases, DNN training time is dominated by *data stall time*: time spent waiting for data to be fetched and pre-processed. We build a tool, DS-Analyzer to precisely measure data stalls using a differential technique, and perform predictive what-if analysis on data stalls. Based on the insights from our analysis, we design and implement three simple but effective techniques in a data-loading library, [CoorDL](https://github.com/msr-fiddle/CoorDL) (built atop [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html)), to
mitigate data stalls. 

[[pdf]](https://www.microsoft.com/en-us/research/publication/analyzing-and-mitigating-data-stalls-in-dnn-training/)  [[slides]]()

## Setup

To run DS-Analyzer and CoorDL you will need a NVIDIA GPU (tested on V100 and 1080Ti) with CUDA 10.1, GPU driver version 418.56, nvidia-docker2, and Python 3. We used the prebuilt NVIDIA docker container [nvcr.io/nvidia/pytorch:19.05-py3](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch/tags) container as the base image, which can be downloaded using,

    docker pull nvcr.io/nvidia/pytorch:19.05-py3
  
We need a few Linux utilities and an updated version of Apex to run our tools. You need to build a new docker image and  install these dependecies using the DockerFile provided [here](docker/DockerFile) as follows

    cd docker
  
    docker build --tag base_image . --file ./Dockerfile

Now run this base docker container using the command

    nvidia-docker run --ipc=host --mount src=/,target=/datadrive/,type=bind -it --rm --network=host --privileged base_image
    
If you want to control the amount of memory and the number of CPUs allocated to the container (as we do in some analysis to vary cache size and the number of CPUs allocated to a job), then pass in additional parameters to the docker run command as follows

    --cpus=16 -m 200g

This ensures that the container uses no more than 16 CPUs and a maximum of 200GB of DRAM for instance.

## Data

Our experiments use the following publicly available large datasets, which can be downloaded from their official repos.

1. ImageNet-1k (~146GB): This is the most widely used image dataset (with 1000 classes) downloadable from [here](http://www.image-net.org/)

2. OpenImages : We use the extended version of OpenImages with 4260 classes (~645GB), which includes the 600 class OpenImages v4 dataset [here](https://storage.googleapis.com/openimages/web/download_v4.html), along with the crowdsourced image subset [here](https://storage.googleapis.com/openimages/web/extended.html)

3. ImageNet-22k (~1.3TB) : This is the full ImageNet dataset with 21841 classes downloadable from the ImageNet website.

4. Free Music Archive (~950GB) : The music clips in this [dataset](https://github.com/mdeff/fma) are first pre-processed offline to convert to a wav format. The utility scripts for this transformation is available [here](fma-utils/)


## Running DS-Analyzer

To measure prep and fetch stalls in a model using DS-Analyzer, a few changes have to be made to the training script to hook some profiling metrics. An complete example of such a modified training script for image classification using popular models like AlexNet, ShuffleNet, Squeezenet, ResNets, and VGGs is shown [here][tool/image_classification/pytorch-imagenet-dali-mp.py]. To evaluate other models, adapt these changes to your training script in accordance with the README instructions in the [tools](tools/) folder.

Before running DS-Analyzer for the first time, install some software dependencies using

        ./prereq.sh
        
You can measure data stalls in ResNet18 when trained across 8 GPUs using 24 CPUs using the following command:
        
        python harness.py --nproc_per_node=8 -j 3 -a resnet18 --prefix results/run1/ image_classification/pytorch-imagenet-dali-mp.py --amp --data-profile --noeval <path_to_dataset_with_train_and_eval_folders>


## Training with CoorDL

First, get CoorDL using

    git clone --recursive https://github.com/msr-fiddle/CoorDL
        
Then, install CoorDL by following the instructions in the [repo](https://github.com/msr-fiddle/CoorDL)
Upon completion, you will have a runnable docker image tagged `nvidia/dali:py36_cu10.run` with CoorDL installed in it.

### Using CoorDL

Training scripts that use DALI as the dataloader can be modified to use CoorDL by simply passing in a few additional parameters to the dataloader which 
the training script parses. In addition to the parameters used by DALI, pass in the values of `cache_size` which represents the maximum number of data items to be cached in memory, and the list of IP and ports of the servers involved, if trained across servers, as shown below

    self.input = ops.FileReader(..., num_nodes=args.nnodes, node_id = args.node_rank, cache_size=args.cache_size, node_port_list=args.node_port_list, node_ip_list=args.node_ip_list)

A full working example of using CoorDL as the dataloader is present [here](tasks/image_classification/pytorch-imagenet-dali-mp.py)

### Example

Training ResNet18 with a per-GPU batch size of 512 on the OpenImages dataset present at <data_path> on a server with 500GB DRAM and 24 CPUs

    cd tasks/image_classification
    
#### 1. Single server training

To train ResNet18 across 8 V100 GPUs in a single server with a per-GPU batch size of 512 on the OpenImages dataset present at <data_path>, use the following commands:
    
With DALI:
    
    python -m torch.distributed.launch --nproc_per_node=8 pytorch-imagenet-dali-mp.py -a resnet18 -b 512 --workers 3 --epochs 3 --amp --classes 4260 <data_path> > stdout.out 2>&1

With CoorDL:

    python -m torch.distributed.launch --nproc_per_node=8 pytorch-imagenet-dali-mp.py -a resnet18 -b 512 --workers 3 --epochs 3 --amp --classes 4260 --cache_size 190000 <data_path> > stdout.out 2>&1


#### 2. Multi server training

To train ResNet18 across two servers, each with 8 V100 GPUs, and a per-GPU batch size of 512 on the OpenImages dataset present at <data_path>, use the following commands:

With DALI:
    
  On server 1: 
  
     python -m launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="x.x.x.x" --master_port=12340 pytorch-imagenet-dali-mp.py -a resnet18 -b 512 --workers 3 --epochs 3 --amp --classes 4260 <data_path> > stdout.out 2>&1
        
  On server 2: 
  
     python -m launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="x.x.x.x" --master_port=12340 pytorch-imagenet-dali-mp.py -a resnet18 -b 512 --workers 3 --epochs 3 --amp --classes 4260 <data_path> > stdout.out 2>&1


With CoorDL:

    
  On server 1 (IP : x.x.x.x): 
  
    cd dist-minio                                             
    ./build.sh                                                       
     mkdir -p /dev/shm/cache                                                          
    ./server 16 5555 &
  
     python -m launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="x.x.x.x" --master_port=12340 pytorch-imagenet-dali-mp.py -a resnet18 -b 512 --workers 3 --epochs 3 --amp --cache_size 190000 --classes 4260 --dist_mint --node_ip_list="x.x.x.x" --node_port_list 5555 --node_ip_list="y.y.y.y" --node_port_list 6666 <data_path> > stdout.out 2>&1
        
  On server 2 (IP : y.y.y.y): 
  
     cd dist-minio                                             
    ./build.sh                                                       
     mkdir -p /dev/shm/cache                                                          
    ./server 16 6666 &
  
     python -m launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="x.x.x.x" --master_port=12340 pytorch-imagenet-dali-mp.py -a resnet18 -b 512 --workers 3 --epochs 3 --amp --cache_size 190000 --classes 4260 --dist_mint --node_ip_list="x.x.x.x" --node_port_list 5555 --node_ip_list="y.y.y.y" --node_port_list 6666 <data_path> > stdout.out 2>&1


#### 3. Hyperparameter Search

To train 8 concurrent ResNet18 jobs on a server with 8 V100 GPUs, and a per-GPU batch size of 512 on the OpenImages dataset present at <data_path>, use the following commands:

With DALI:
          
     python -O hyperparam_imagenet_dali_mp.py -a resnet18 -j 3 -b 512 --epochs 3 --dist --num_jobs 8 --amp --classes 4260 <data_path> > stdout.out 2>&1
          
With CoorDL:

     python -O hyperparam_imagenet_dali_mp.py -a resnet18 -j 3 -b 512 --epochs 3 --dist --num_jobs 8 --amp --dali_cpu --unified --pin --classes 4260 <data_path> > stdout.out 2>&1


---

This project has adopted the Microsoft Open Source Code of Conduct. For more information see the Code of Conduct FAQ.
License
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT license." 
