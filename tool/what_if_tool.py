"""
This is a test harness for profiling DNN training scripts 
to answer what-if questions on data stalls during training.

**How does this framework work?:**

This framework profiles the workload and collects statistics such as
    1. Avg per-iteration time ( MAX INGESTION RATE of the model)
    2. Avg per-iteration pre-processing stalls 
    3. Avg per-iteration data fetch stalls
    4. Optimal number of CPU per GPU to mask pre-processing stalls
    5. Avg disk throughput
    6. Available network bandwidth
    7. Optimal cache size

Our framework does this in a series of steps:

1. Train the model for a few iterations with synthetic data
   - Synchronize after cuda memcpy
   - Synchronize at iteration boundaries
   - Profiles the memcpy time and actual GPU time per iteration

2. Train the model for a few iterations with actual data
   with a cold cache
   - Synchronize after data get
   - SYnchronize at iteration boundaries
   - Profiles the actual GPU time and pre-processing + data fetch time
   
3. Train the model for a few iterations with actual data
   that is fully cached
   - Synchronize after data get
   - Synchronize at iteration boundaries
   - Profiles the actual GPU time and pre-processing time

MAX INGESTION RATE = Num_minibatches * Size per minibatch / (1)
PRE PROCESING RATE = Num_minibatches * Size per minibatch / (3)
DATA FETCH RATE    = Num_minibatches * Size per minibatch / ((2) - (3))
  
"""


import sys
import subprocess
import os
import utils
from argparse import ArgumentParser, REMAINDER
from synthetic_data import get_shared_image_classification_tensors
from utils import aggregate_run1_maps, print_as_table, print_header
import multiprocessing
import json

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch data stall profiler")

    # Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")


    # profiling
    parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                         help='number of data loading workers')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                         help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                         metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')
    parser.add_argument('--synthetic', action='store_true',
                         help='Use synthetic dataset')
    parser.add_argument('--data-profile', action='store_true',
                         help='Set profiler on')  
    parser.add_argument('--precreate', action='store_true')
    parser.add_argument('--use_precreate', action='store_true')
    parser.add_argument("--classes", default=1000, type=int)
    parser.add_argument("--tensor_path", default="./train", type=str)
    parser.add_argument("--num_minibatches", default=50, type=int)
    parser.add_argument("--path", default="./", type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument('-q', '--question', default="cache", type=str)


    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()

args = parse_args()

def analyze_cache():
    print("In analyze cache")
    if args.model_path is not None:
        _analyze_cache(args.model_path)
    else:
        for job_path in args.jobs:
          print("Analysis for jon {}".format(job_path))
          #Must analyze for each GPU and CPU combination seperately
          gpu_paths = [os.path.join(job_path, o) for o in os.listdir(job_path) if os.path.isdir(os.path.join(job_path,o))]
          for gpu_path in gpu_paths:
            cpu_paths = [os.path.join(gpu_path, o) for o in os.listdir(gpu_path) if os.path.isdir(os.path.join(gpu_path,o))]
            for cpu_path in cpu_paths:
                print(cpu_path)
                model_path = cpu_path + "/MODEL.json"
                _analyze_cache(model_path)

def _analyze_cache(model_path):
                model = {}
                with open(model_path, 'r') as mf:
                    model = json.load(mf)
                # Have the model now. Calculate speeds for different cache sizes
                max_speed = model["SPEED_INGESTION"]
                cached_speed = model["SPEED_CACHED"]
                disk_thr = model["DISK_THR"]
                disk_bw = 525
                mem_thr = model["MEM_THR"]
                dataset_size = model["AVG_SAMPLE_SIZE"]
                total_samples = model["TOTAL_SAMPLES"]

                speed_map = {}
                for cache_percent in range(0, 100, 5):
                    cache_size = cache_percent/100*dataset_size
                    disk_fetch_size = dataset_size - cache_size
                    time_to_cache = cache_size/mem_thr
                    time_to_disk = disk_fetch_size*1024/disk_bw
                    total_time = time_to_cache + time_to_disk
                    avg_sample_size = dataset_size*1024*1024 / total_samples
                    #effective_store_thr = dataset_size*1024*1024/total_time
                    effective_store_thr = dataset_size*1024*1024/total_time/avg_sample_size
                    speed_map[cache_percent] = effective_store_thr
                keys = speed_map.keys()
                values = speed_map.values()
                print("Max achievable speed = {} samples/s".format(int(cached_speed))) 
                for key in keys:
                    print("{:<5}".format(int(key)), end=" ")
                print("\n")
                for val in values:
                    print("{:<5}".format(int(val)), end = " ")
                print("\n")            
                print("-"*100)            
        
        


def question(qname):
    switch = {
        "cache" : analyze_cache
        # Add more analysis questions
    }

    func = switch.get(qname, lambda: "Invalid analysis option")
    func()

def main():
    """
    args.path is the path to the model being analyzed.
    The expected heirarchy is :
     <model>
     |--jobs-<count_per_node>
        |--gpus-<count_per_job>
           |--cpus-<count_per_gpu>
               |--MODEL.json
           
    """
    args.jobs = [os.path.join(args.path, o) for o in os.listdir(args.path) if os.path.isdir(os.path.join(args.path,o))]
    print(args.jobs)
    question(args.question)

   
if __name__ == "__main__":
    main()
