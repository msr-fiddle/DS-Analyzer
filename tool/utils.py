import os
import sys
import statistics
import csv

def str2bool(v):    
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def init_maps(dest, source):
    keys = source.keys()
    for key in keys:
        dest[key] = 0
    return keys, dest


"""
  Gets a list of maps, aggregates each key
  and returns one aggregate map 
  with the same keys
"""
def aggregate_run1_maps(list_of_map):
    num_maps = len(list_of_map)
    stdev_map = {}
    mean_map = {}
    keys, mean_map = init_maps(mean_map, list_of_map[0])
    keys, stdev_map = init_maps(stdev_map, list_of_map[0])

    for key in keys:
        val_list = []
        for i in range(0, num_maps):
            val_list.append(list_of_map[i][key])
        mean = statistics.mean(val_list) 
        stddev = statistics.pstdev(val_list, mean)
        stdev_map[key] = stddev
        mean_map[key] = mean
    return mean_map, stdev_map
   

def print_as_table(prof_map, key=None):
    if not isinstance(prof_map,dict):
        if key == "MEM_THR":
            print("{:<20} {:.2f} GB/s".format(key, prof_map))
        elif key == "DISK_THR":
            print("{:<20} {:.2f} MB/s".format(key, prof_map))
        elif "SPEED" in key:
            print("{:<20} {:.2f} samples/s".format(key, prof_map))
        elif "SIZE" in key:
            print("{:<20} {} KB".format(key, prof_map))
        else:
            print("{:<20} {}".format(key, prof_map))
        return

    print("="*30)
    print("{:<10} {:<10}".format('Metric', 'Value'))
    print("-"*30)
    for key, value in prof_map.items():
        if key == "SPEED":
            print("{:<10} {:.2f} samples/s".format(key, value))
        elif key in ["READ", "RECV", "SEND"]:
            print("{:<10} {:.2f} MB/s".format(key, value))
        elif key in ["MEM", "CACHE", "PCACHE"]:
            print("{:<10} {:.2f} GB".format(key, value))
        elif key == "CPU":
            print("{:<10} {:.2f} %".format(key, value))
        elif key in ["MEMCPY", "DATA", "COMPUTE", "TRAIN"]:
            print("{:<10} {:.2f} s".format(key, value))
        else: 
            print("{:<10} {:<10}".format(key, value))
    print("="*30)

def print_header(args):
    print("="*30)
    print("DATA STALL PROFILER")
    print("-"*30)
    print("{:<10} {:<10}".format('Model', args.arch))
    print("{:<10} {:<10}".format('GPUs', args.nnodes*args.nproc_per_node))
    print("-"*30)

def print_all(prof_maps, expand=True):
    print("_"*40)
    print(" Statistics collected so far..")
    print("-"*40)
    for key, prof_map in prof_maps.items():
         if not expand and not isinstance(prof_map,dict):
              print_as_table(prof_map, key=key)
         elif expand:
              print_as_table(prof_map, key=key)
    print("_"*40)
    print("\n")
    
   

def move_logs(log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    cmd = 'mv *.log ' + log_path
    os.system(cmd)
    cmd = 'mv *.csv ' + log_path
    os.system(cmd)
    cmd = 'mv *.json ' + log_path
    os.system(cmd)
    cmd = 'mv *.out ' + log_path
    os.system(cmd)

#Format : ----CPU---------------------,--Disk-----,--N/w-----,--------Memory----------
#Format : usr, sys, idl, wai, hiq, siq, read, writ, recv, send, used, buff, cach, free
def parseDstat(fname):
    csvfile = open(fname, "r")
    idle_list = []
    wai_list = []
    read_list = []
    write_list = []
    recv_list = []
    send_list = []
    for i in range(6):
        next(csvfile)
    reader = csv.DictReader(csvfile)
    header = reader.fieldnames
    for row in reader:
        idle_list.append(float(row["idl"]))
        wai_list.append(float(row["wai"]))
        read_list.append(float(row["read"])/1024/1024) #in MB
        write_list.append(float(row["writ"])/1024/1024)
        recv_list.append(float(row["recv"])/1024/1024)
        send_list.append(float(row["send"])/1024/1024)

    mean_idle = statistics.mean(idle_list)
    mean_wait = statistics.mean(wai_list)
    mean_read = statistics.median_grouped(read_list)
    mean_write = statistics.mean(write_list)
    mean_recv = statistics.mean(recv_list)
    mean_send = statistics.mean(send_list)
    #total_read = sum(read_list)
    #total_recv = sum(recv_list)
    #total_send = sum(send_list)
    #print(mean_read)
    return (mean_idle, mean_wait, mean_read, mean_write, mean_recv, mean_send)

#Format : None, total, used, free, shared, cache, avail    
def parseFree(fname):
    csvfile = open(fname, "r")   
    reader = csv.DictReader(csvfile) 
    header = reader.fieldnames 
    total_list = []
    pmem_list = []   #Process working size
    shm_list = []    #Shared memory
    page_cache = []  #Page cache
    line = 0
    start_cache = start_used = start_shm = 0
    for row in reader: 
        if line == 0:
            start_cache = float(row["cache"])
            start_used = float(row["used"])
            start_shm = float(row["shared"])
        else:
            pmem_list.append(float(row["used"]) - start_used)
            page_cache.append(float(row["cache"]) - start_cache - (float(row["shared"]) - start_shm) )
            shm_list.append(float(row["shared"]) - start_shm)
            total = float(row["used"]) - start_used + float(row["cache"]) - start_cache
            total_list.append(total)
        line += 1

    max_pmem = max(pmem_list)
    max_shm = max(shm_list)
    max_page_cache = max(page_cache)
    max_total = max(total_list)
    return (max_pmem, max_shm, max_page_cache, max_total)


def start_resource_profiling():
    os.system("dstat -cdnm --output all-utils.csv 2>&1 >> redirect-dstat.log &")
    os.system("./free.sh &")

def stop_resource_profiling():
    os.system("pkill -f dstat")
    os.system("pkill -f free")
    os.system("./parseFree.sh free.out")
    res = parseDstat('all-utils.csv')
    res_free = parseFree('free.csv')
    return res, res_free

def clear_cache():
    os.system("echo 3 > /proc/sys/vm/drop_caches")
    print("Cleared Page Cache...")
