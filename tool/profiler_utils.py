import os
import sys
import time
from tqdm import tqdm
import json
import logging
#from instant import inline
from colorama import Fore

class SuppressStream(object): 

    def __init__(self, stream=sys.stderr, fname=None):
        self.orig_stream_fileno = stream.fileno()
        self.fname = fname

    def open(self):
        self.orig_stream_dup = os.dup(self.orig_stream_fileno)
        self.orig_stdout = os.fdopen(self.orig_stream_dup, 'w')
        if self.fname is None:
            self.devnull = open(os.devnull, 'w')
        else:
            self.devnull = open(self.fname, 'w')
        os.dup2(self.devnull.fileno(), self.orig_stream_fileno)
        return self.orig_stdout

    def close(self):
        os.close(self.orig_stream_fileno)
        os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
        os.close(self.orig_stream_dup)
        self.devnull.close()


class DataStallProfiler():
    def __init__(self, args):
        self.args = args
        if self.args.suffix is None:
            self.args.suffix = str(self.args.local_rank)
        self.log_outfile = 'stdoutlog-' + self.args.suffix + '.log'
        self.err_outfile = 'stderrlog-' + self.args.suffix + '.log'
        pout_file = 'profile-'  + self.args.suffix + '.json'
        self.pout = open(pout_file, "w")

        time_outfile = 'time-' + self.args.suffix + '.csv'
        self.time_logger = open(time_outfile, 'w')
        if self.args.synthetic:
            self.time_logger.write("Iter, Memcpy Time,  Data time, Compute time \n")
        else:
            self.time_logger.write("Iter, Data time, Compute time \n")
        self.data_time = 0
        self.memcpy_time = 0
        self.compute_time = 0
        self.total_data_time = 0
        self.total_compute_time = 0
        self.total_memcpy_time = 0
        self.active = False
        self.active_sub = False
        self.iter = 0
        self.batch_count = 0
        self.num_samples = 0 
        self.id = self.args.local_rank
        self.running_time = time.time()
        self.train_time = time.time()
        self.warmup = 5

        self.stream = SuppressStream(sys.stdout, self.log_outfile)       
        self.stream_err = SuppressStream(sys.stderr, self.err_outfile)     
        self.stream.open()
        bar_fhandle = self.stream_err.open()
        #sys.stdout = open(self.log_outfile, "w")
        if self.id == 0:
            self.bar = tqdm(total=self.args.num_minibatches, 
                            file=bar_fhandle, 
                            desc="Iteration Progress",
                            dynamic_ncols=True,
                            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET))
 

    def stop_profiler(self):
        if self.id == 0:
            self.bar.close()
        self.stream.close()
        self.stream_err.close()
        self.time_logger.close()
        #sys.stdout.close()
        #sys.stdout = sys.__stdout__
        self.running_time = time.time() - self.running_time
        self.train_time = time.time() - self.train_time
        #if True:
        """
            if self.args.synthetic:
                print("[{}] Total Memcpy Time = {}".format(self.id, self.total_memcpy_time))
            print("[{}] Total Data Time = {}".format(self.id, self.total_data_time - self.total_memcpy_time))
            print("[{}] Total Compute Time = {}".format(self.id, self.total_compute_time))
            print("[{}] Total Training Time = {}".format(self.id, self.train_time))
            print("[{}] Total Running Time = {}".format(self.id, self.running_time))
            
            #num_samples = self.args.batch_size* (self.args.num_minibatches - self.warmup -1) * self.args.world_size

            total_time = self.total_compute_time
            print("[{}] MAX INGESTION RATE = {:0.2f} samples/s".format(self.id, self.num_samples/total_time ))
            print("[{}] Total samples = {}".format(self.id, self.num_samples))
        """
        
        # Write out a json file of stats from this run
        stats = {}
        stats["MEMCPY"] = self.total_memcpy_time
        stats["DATA"] = self.total_data_time
        stats["COMPUTE"] = self.total_compute_time
        stats["TRAIN"] = self.train_time
        stats["BATCHES"] = self.batch_count
        stats["SAMPLES"] = self.num_samples
        json.dump(stats, self.pout)
        self.pout.close()

    def start_memcpy_tick(self):
        if self.iter < self.warmup:
            return
        self.active_sub = True
        self.memcpy_time = time.time() 
        
        
    def stop_memcpy_tick(self):
        if self.iter < self.warmup:
            return
        if self.active:
            self.memcpy_time = time.time() - self.memcpy_time
            self.total_memcpy_time += self.memcpy_time
            self.active_sub = False
        else:
            print("ERR in iter {} MEMCPY".format(self.iter))
            raise Exception("Timer stopeed without starting")


    def start_data_tick(self):
        self.iter += 1
        if self.iter < self.warmup:
            return
        self.active = True
        self.data_time = time.time()       
        if self.iter == self.warmup: 
            self.train_time = time.time()

    def stop_data_tick(self):
        if self.iter < self.warmup:
            return
        if self.active:
            self.data_time = time.time() - self.data_time
            self.active = False
            self.total_data_time += self.data_time
        else:
            print("ERR in iter {} DATA".format(self.iter))
            raise Exception("Timer stopeed without starting")

    def start_compute_tick(self):
        if self.iter < self.warmup:
            return
        self.active = True
        self.compute_time = time.time()        

    def stop_compute_tick(self):
        if self.id == 0:
            self.bar.update(1)
        if self.iter < self.warmup:
            return
        if self.active:
            self.compute_time = time.time() - self.compute_time
            self.num_samples += (self.args.world_size * self.args.batch_size) 
            self.batch_count += 1
            self.active = False
            #Write both data and compute time to file
            if not self.args.synthetic:
                line = str(self.iter) + "," + str(self.data_time) + "," + str(self.compute_time) + "\n"
            else:
                line = str(self.iter) + "," + str(self.memcpy_time) + "," + str(self.data_time) + "," + str(self.compute_time) + "\n"
            self.time_logger.write(line)
            self.total_compute_time += self.compute_time
        else:
            print("ERR in iter {} COMP".format(self.iter))
            raise Exception("Timer stopeed without starting")

