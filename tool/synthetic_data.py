import os
import time
import sys
import torch
import threading
import multiprocessing

class SyntheticClassificationIterator(object):    
    def __init__(self, args):  
        self.args = args       
        self.iter_num = -1   
        self.dev = self.args.local_rank 

        if self.args.precreate:
            self.tensor_bank = load_tensors(self.args.tensor_path, self.args.num_minibatches)
        else:
            self.tensor_bank = get_image_classification_tensors(self.args.batch_size, self.args.num_minibatches, self.args.classes)    
        print("Got {} tensors".format(len(self.tensor_bank.keys()))) 
        if len(self.tensor_bank.keys()) == 0:
            raise Exception("Could not precreate tensors!")   
    @property
    def _size(self):
        return self.args.num_minibatches*self.args.batch_size

    def __iter__(self):
        return self
 
    def __next__(self):     
        self.iter_num += 1         
        if self.iter_num < self.args.num_minibatches:   
            self.args.dprof.start_memcpy_tick()  
            images, target = self.tensor_bank[self.iter_num] 
            images = images.cuda(self.dev) 
            target = target.cuda(self.dev) 
            self.args.dprof.stop_memcpy_tick()  
            return images, target   
        else:        
            raise StopIteration    

    def next(self):      
        return self.__next__()   

def load_tensors(path, total):
    s = time.time()   
    tensor_bank = {}
    th = []
    for i in range(0,5):
        th.append(threading.Thread(target=_load, args=(i*int(total/5), int(total/5), path, tensor_bank)))
        th[i].start()
    
    for i in range(0,5):
        th[i].join()
    print("Loaded {} tensors in {} s".format(len(tensor_bank.keys()), time.time() - s)) 
    return tensor_bank

def _load(start, count, path, tensor_bank):
    for i in range(start, start+count):
        img_name = path + "/image-" + str(i) + ".pt"
        label_name = path + "/label-" + str(i) + ".pt"
        image = torch.load(img_name)
        label = torch.load(label_name)
        tensor_bank[i] = (image, label)


def get_shared_image_classification_tensors(batch_size, iters, start, num_classes=1000, path="./train"): 
    print("Pre-populating train tensors ...")   
    s = time.time()   
    for i in range(start, start+iters):     
        img_name = path + "/image-" + str(i) + ".pt"
        label_name = path + "/label-" + str(i) + ".pt"
        img = getRandImgClassificationTensor(batch_size) 
        target = getRandTargetClassificationTensor(batch_size, num_classes) 
        torch.save(img, img_name)
        torch.save(target, label_name)
    print("Created {} tensors in {} s".format(iters, time.time() - s)) 


 
def get_image_classification_tensors(batch_size, iters, num_classes=1000):  
    print("Pre-populating train tensors ...")   
    tensor_bank={}     
    s = time.time()   
    for i in range(0, iters):     
    #for i in range(start, start+iters):     
        img = getRandImgClassificationTensor(batch_size) 
        target = getRandTargetClassificationTensor(batch_size, num_classes) 
        tensor_bank[i] = (img,target)          
    print("Created {} tensors in {} s".format(iters, time.time() - s)) 
    return tensor_bank 

def getRandImgClassificationTensor(batchsize):  
    return torch.randn(batchsize, 3, 224, 224)   

def getRandTargetClassificationTensor(batchsize, num_classes):  
    return torch.randint(0, num_classes, (batchsize,), dtype=torch.long)  
 
