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
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")
    parser.add_argument("--use_env", default=False, action="store_true",
                        help="Use environment variable to pass "
                             "'local rank'. For legacy reasons, the default value is False. "
                             "If set to True, the script will not pass "
                             "--local_rank as argument, and will instead set LOCAL_RANK.")

    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")


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
    parser.add_argument("--full_epoch", default=False, action='store_true')
    parser.add_argument("--resume_json", default=None, type=str)
    parser.add_argument("--prefix", default="", type=str)


    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()

args = parse_args()

def run_synthetic(): 
    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)


    if args.precreate:
        print("Precreating tensors in {}".format(args.tensor_path))
        if not os.path.exists(args.tensor_path):
            os.makedirs(args.tensor_path)
        procs = []
        for i in range(5):
            procs.append(multiprocessing.Process(target=get_shared_image_classification_tensors, args=(args.batch_size, int(args.num_minibatches/5),  i*int(args.num_minibatches/5), args.classes, args.tensor_path)))
            procs[i].start()

        for i in range(5):
            procs[i].join()
        
        args.use_precreate = True


    processes = []

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        if args.use_env:
            cmd = [sys.executable, "-u",
                   args.training_script] + args.training_script_args
        elif args.use_precreate:
            cmd = [sys.executable,
                   "-u",
                   args.training_script,
                   "--local_rank={}".format(local_rank), 
                   "--nnodes={}".format(args.nnodes),
                   "--node_rank={}".format(args.node_rank),
                   "--batch-size={}".format(args.batch_size),
                   "--workers={}".format(args.workers),
                   "--classes={}".format(args.classes),
                   "--num_minibatches={}".format(args.num_minibatches),
                   "--precreate",
                   "--tensor_path={}".format(args.tensor_path),
                   "--arch={}".format(args.arch),
                   "--synthetic",
                   "--epochs={}".format(args.epochs)] + args.training_script_args

        else:
            cmd = [sys.executable,
                   "-u",
                   args.training_script,
                   "--local_rank={}".format(local_rank), 
                   "--nnodes={}".format(args.nnodes),
                   "--node_rank={}".format(args.node_rank),
                   "--batch-size={}".format(args.batch_size),
                   "--workers={}".format(args.workers),
                   "--classes={}".format(args.classes),
                   "--num_minibatches={}".format(args.num_minibatches),
                   "--arch={}".format(args.arch),
                   "--synthetic",
                   "--epochs={}".format(args.epochs)] + args.training_script_args

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=process.args)
    log_path = os.getcwd() + "/" + args.prefix + "/" +  args.arch + "/jobs-1"  + "/gpus-" + str(dist_world_size) + "/cpus-" + str(args.workers) +  "/run1-synthetic/"

    
    utils.move_logs(log_path)
    print("FINISHED STEP 1 : SYNTHETIC WORKLOAD")
    return log_path


def run_with_data(cached=False):
    dist_world_size = args.nproc_per_node * args.nnodes
    if not cached: 
        log_path = os.getcwd() + "/" + args.prefix + "/" + args.arch + "/jobs-1" + "/gpus-" + str(dist_world_size) +  "/cpus-" + str(args.workers) + "/run2-fetch-preprocess/"
    else:
        log_path = os.getcwd() + "/" +  args.prefix + "/" + args.arch + "/jobs-1"+ "/gpus-" + str(dist_world_size) + "/cpus-" + str(args.workers) + "/run3-preprocess/"
      
    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)
    processes = []

    utils.start_resource_profiling()

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        if args.use_env:
            cmd = [sys.executable, "-u",
                   args.training_script] + args.training_script_args
        elif not args.full_epoch:
            cmd = [sys.executable,
                   "-u",
                   args.training_script,
                   "--local_rank={}".format(local_rank), 
                   "--nnodes={}".format(args.nnodes),
                   "--node_rank={}".format(args.node_rank),
                   "--batch-size={}".format(args.batch_size),
                   "--workers={}".format(args.workers),
                   "--classes={}".format(args.classes),
                   "--num_minibatches={}".format(args.num_minibatches*2),
                   "--arch={}".format(args.arch),
                   "--epochs={}".format(args.epochs)] + args.training_script_args
        else:
            cmd = [sys.executable,
                   "-u",
                   args.training_script,
                   "--local_rank={}".format(local_rank), 
                   "--nnodes={}".format(args.nnodes),
                   "--node_rank={}".format(args.node_rank),
                   "--batch-size={}".format(args.batch_size),
                   "--workers={}".format(args.workers),
                   "--classes={}".format(args.classes),
                   "--num_minibatches={}".format(args.num_minibatches*2),
                   "--full_epoch ",
                   "--arch={}".format(args.arch),
                   "--epochs={}".format(args.epochs)] + args.training_script_args

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=process.args)

    res_dstat, res_free = utils.stop_resource_profiling()
    utils.move_logs(log_path)
    if not cached:
        print("FINISHED STEP 2 : PREPROCESS + FETCH ")
    else:
        print("FINISHED STEP 3 : PREPROCESS ONLY")
    return log_path, res_dstat, res_free


def run_mem_test(cmd):
   process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
   (output,err)=process.communicate()
   exit_code = process.wait()
   return exit_code

def get_dataset_stats(dir_path):
   train_path = dir_path + "/train/"
   cmd = "du -sh " + train_path
   process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
   (output,err)=process.communicate()
   exit_code = process.wait()
   size = output.decode('utf-8').split()[0][:-1]
   metric = output.decode('utf-8').split()[0][-1]
   if str(metric) == "T":
       size = int(float(size)*1024)
 
   cmd = "find " + train_path + " -type f | wc -l"
   process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
   (output,err)=process.communicate()
   exit_code = process.wait()
   samples = output.decode('utf-8').split()[0]

   return size, samples


def main():
    print_header(args)
    num_gpu = args.nproc_per_node * args.nnodes  
    args.stats = {}
    resume = False
    if not (args.resume_json is None):
        resume = True
        print("Resuming from existing profile stats at {}".format(args.resume_json))
        if not os.path.exists(args.resume_json):
            print("Incorrect resume stat path")
            sys.exit(1)
        with open(args.resume_json, 'r') as jf:
            args.stats = json.load(jf)

    final_log_path = os.getcwd() + "/" + args.prefix + "/" + args.arch + "/jobs-1" + "/gpus-" + str(num_gpu) +  "/cpus-" + str(args.workers) + "/"

    # Stage 1 : Run with synthetic dataset
    if resume and 'RUN1' in args.stats:
        print_as_table(args.stats["RUN1"])
        print("STEP 1 already done. Continuing to step 2")

    else:
        log_path = run_synthetic()
        print("Parsing Step 1 results ...")
    
        """
         JSON is of the following format : All times are total for BATCHES num batches
         1. MEMCPY - Time to memcpy the synthetic tensors to GPU DRAM
         2. DATA - Total time for a batch to be ready at GPU - Includes the memcpy time 
         3. COMPUTE - Total GPU computation time the batch   
         4. TRAIN - Total training time, sum of data and compute
         5. BATCHES - The number of batches whose collective times are shown above
         6. SAMPLES - Total samples used in this profiling phase
         All these numbers exclude any warmup batches specified 
        """
        run1_stats = []
        for i in range(0,num_gpu):
            json_file =  log_path + 'profile-' + str(i) + '.json'
            run1_stats.append(json.load(open(json_file)))
            
        if len(run1_stats) != num_gpu:
            print("Something went wrong in run1")
            sys.exit(1)
    
        args.stats["RUN1"], stddev_map = aggregate_run1_maps(run1_stats)
        args.stats["RUN1"]["SPEED"] = args.stats["RUN1"]["SAMPLES"]/args.stats["RUN1"]["COMPUTE"]
        args.stats["SPEED_INGESTION"] = args.stats["RUN1"]["SPEED"]

        for value in list(stddev_map.values()):
            if value > 1:
                print("High STDDEV in values. Run for more minibatches for stable results")
                #sys.exit(1)
        
        print_as_table(args.stats["RUN1"])


    # Stage 2 : Run with both fetch and pre-processing on 
    if resume and 'RUN2' in args.stats:
        print_as_table(args.stats["RUN2"])
        print("STEP 2 already done. Continuing to step 3\n")
    else:
        #Drop cache here
        utils.clear_cache()

        log_path, res_dstat, res_free = run_with_data()    
        idle, wait, read, write, recv, send= res_dstat
        pmem, shm,page_cache, total = res_free

        print("\nParsing Step 2 results ...")
        run2_stats = []
        for i in range(0,num_gpu):
            json_file =  log_path + 'profile-' + str(i) + '.json'
            run2_stats.append(json.load(open(json_file)))
            
        if len(run2_stats) != num_gpu:
            print("Something went wrong in run1")
            sys.exit(1)
    
        args.stats["RUN2"], stddev_map = aggregate_run1_maps(run2_stats)
        args.stats["RUN2"]["SPEED"] = args.stats["RUN2"]["SAMPLES"]/args.stats["RUN2"]["TRAIN"]
        args.stats["RUN2"]["RECV"] = recv
        args.stats["RUN2"]["SEND"] = send
        args.stats["RUN2"]["READ"] = read
        args.stats["RUN2"]["CPU"] = 100 - idle
        args.stats["RUN2"]["MEM"] = pmem + shm
        args.stats["RUN2"]["PCACHE"] = page_cache

        args.stats["DISK_THR"] = args.stats["RUN2"]["READ"]
        args.stats["SPEED_DISK"] = args.stats["RUN2"]["SPEED"]

        print_as_table(args.stats["RUN2"])


    # Stage 3 : Run with only pre-processing
    if resume and 'RUN3' in args.stats:
        print_as_table(args.stats["RUN3"])
        print("STEP 3 already done. Continuing to step 4\n")
    else:
        log_path, res_dstat, res_free = run_with_data(cached = True)    
        idle, wait, read, write, recv, send = res_dstat
        pmem, shm,page_cache, total = res_free
        print("\nParsing Step 3 results ...")
        run3_stats = []
        for i in range(0,num_gpu):
            json_file =  log_path + 'profile-' + str(i) + '.json'
            run3_stats.append(json.load(open(json_file)))
            
        if len(run3_stats) != num_gpu:
            print("Something went wrong in run1")
            sys.exit(1)
    
        args.stats["RUN3"], stddev_map = aggregate_run1_maps(run3_stats)
        args.stats["RUN3"]["SPEED"] = args.stats["RUN3"]["SAMPLES"]/args.stats["RUN3"]["TRAIN"]
        args.stats["RUN3"]["RECV"] = recv
        args.stats["RUN3"]["SEND"] = send
        args.stats["RUN3"]["READ"] = read
        args.stats["RUN3"]["CPU"] = 100 - idle
        args.stats["RUN3"]["MEM"] = pmem + shm
        args.stats["RUN3"]["PCACHE"] = page_cache

        args.stats["SPEED_CACHED"] = args.stats["RUN3"]["SPEED"]

        print_as_table(args.stats["RUN3"])



    # Stage 4 : Run memory throughput test
    if resume and 'MEM_THR' in args.stats:
        print("Memory bandwidth estimation already done. Continuing to step 5\n")
    else:
        cmd = "./memtest " + str(args.workers*args.nproc_per_node) 
        thr = run_mem_test(cmd)
        args.stats["MEM_THR"] = thr


    if resume and 'AVG_SAMPLE_SIZE' in args.stats:
        print("Datasets statistics already collected. Continuing to step 6\n")
    else:
        size, total_samples =  get_dataset_stats(args.training_script_args[-1])
        args.stats["AVG_SAMPLE_SIZE"] = int(size)
        args.stats["TOTAL_SAMPLES"] = int(total_samples)
        

    # Finally dump all stats to a json which can be querried later
    json_outfile = final_log_path + 'MODEL.json'
    with open(json_outfile, 'w') as jf:
        json.dump(args.stats, jf)

    utils.print_all(args.stats, expand=False)
    print("Done writing final JSON : {}".format(json_outfile))

if __name__ == "__main__":
    main()
