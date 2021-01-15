from timeit import default_timer as timer
import sys
import argparse

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

from nvidia.dali.plugin.pytorch import DALIClassificationIterator


def parse():
    parser = argparse.ArgumentParser(description='DALI pre-processing Test')
    parser.add_argument('--data',  default='./', type=str)
    parser.add_argument('--threads',  default=3, type=int)
    parser.add_argument('--dali_cpu',  default=False, action='store_true')
    parser.add_argument('--memcpy',  default=False, action='store_true')
    parser.add_argument('--cuda',  default=False, action='store_true')
    args = parser.parse_args()
    return args

test_batch_size = 512


class FetchPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir="./", dali_cpu=False):
        super(FetchPipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = data_dir, random_shuffle = True)

    def define_graph(self):
        jpegs, labels = self.input()
        return (jpegs, labels)


class DecodePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir="./", dali_cpu=False):
        super(DecodePipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = data_dir, random_shuffle = True)
        if dali_cpu:
            decoder_device = "cpu"
        else:
            decoder_device = "mixed"
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoder(device = decoder_device, output_type = types.RGB,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        return (images, labels)


class DecodeAndCropPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir="./", dali_cpu=False):
        super(DecodeAndCropPipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = data_dir, random_shuffle = True)
        if dali_cpu:
            decoder_device = "cpu"
        else:
            decoder_device = "mixed"
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoder(device = decoder_device, output_type = types.RGB,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding)
        dali_device = "cpu" if dali_cpu else "gpu"
        self.res = ops.RandomResizedCrop(device=dali_device,
                size=224, 
                interp_type=types.INTERP_TRIANGULAR,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100)
				
    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        images = self.res(images)
        return (images, labels)



class DecodeFullSplitPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir="./", dali_cpu=False):
        super(DecodeFullSplitPipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = data_dir, random_shuffle = True)
        if dali_cpu:
            decoder_device = "cpu"
        else:
            decoder_device = "mixed"
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoder(device = decoder_device, output_type = types.RGB,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding)
        dali_device = "cpu" if dali_cpu else "gpu"
        self.res = ops.RandomResizedCrop(device=dali_device,
                size=224, 
                interp_type=types.INTERP_TRIANGULAR,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100)
        print("Device = {}".format(dali_device))
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                output_dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(224, 224),
                mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return (output, labels)


class DecodeSplitBestPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir="./", dali_cpu=False):
        super(DecodeSplitBestPipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = data_dir, random_shuffle = True)
        if dali_cpu:
            decoder_device = "cpu"
        else:
            decoder_device = "mixed"
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding)
        dali_device = "cpu" if dali_cpu else "gpu"
        self.gpu = not dali_cpu
        self.res = ops.RandomResizedCrop(device=dali_device,
                size=224, 
                interp_type=types.INTERP_TRIANGULAR,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100)
        print("Device = {}".format(dali_device))
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                output_dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(224, 224),
                mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        if self.gpu:  
            images = self.res(images.gpu())
        else:
            images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return (output, labels)



class DecodeCropPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir="./", dali_cpu=False):
        super(DecodeCropPipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = data_dir, random_shuffle = True)
        if dali_cpu:
            decoder_device = "cpu"
        else:
            decoder_device = "mixed"
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device = decoder_device, output_type = types.RGB,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        return (images, labels)



class DecodeCropResizePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir="./", dali_cpu=False):
        super(DecodeCropResizePipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = data_dir, random_shuffle = True)
        if dali_cpu:
            decoder_device = "cpu"
        else:
            decoder_device = "mixed"
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        dali_device = "cpu" if dali_cpu else "gpu"
        self.decode = ops.ImageDecoderRandomCrop(device = decoder_device, output_type = types.RGB,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100)
        self.res = ops.Resize(device=dali_device,
                resize_x=224,
                resize_y=224,
                interp_type=types.INTERP_TRIANGULAR)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        images = self.res(images)
        return (images, labels)



class DecodeFullPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir="./", dali_cpu=False):
        super(DecodeFullPipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = data_dir, random_shuffle = True)
        if dali_cpu:
            decoder_device = "cpu"
        else:
            decoder_device = "mixed"
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        dali_device = "cpu" if dali_cpu else "gpu"
        self.decode = ops.ImageDecoderRandomCrop(device = decoder_device, output_type = types.RGB,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100)
        self.res = ops.Resize(device=dali_device,
                resize_x=224,
                resize_y=224,
                interp_type=types.INTERP_TRIANGULAR)
        print("Device = {}".format(dali_device))
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                output_dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(224, 224),
                mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return (output, labels)



def speedtest(pipeclass, batch, n_threads, data='./', dali_cpu=False, memcpy=False, cuda=False):
    pipe = pipeclass(batch, n_threads, 0, data_dir=data, dali_cpu=dali_cpu)
    pipe.build()
    if not memcpy:
        for i in range(5):
            pipe.run()

    n_test=5000
    num_processed = 0
    t_start = timer()
    if not memcpy:
        for i in range(n_test):
            images,label = pipe.run()
            num_processed += len(images)
    else:
        loader = DALIClassificationIterator(pipe, size=1000000)
        for i in range(n_test):
            out = loader.next()
            images = out[0]["data"]
            if cuda:
                images = images.cuda()
    t = timer() - t_start
    print("Speed {}: {:.2f} imgs/s, time={:.2f}s, Images={}".format(type(pipe).__name__, (n_test * batch)/t, t, num_processed))
    print("---"*20)

def main():
    args = parse() 
    speedtest(FetchPipeline, test_batch_size, args.threads, args.data, args.dali_cpu)
    speedtest(DecodeCropPipeline, test_batch_size, args.threads, args.data, args.dali_cpu)
    speedtest(DecodeCropResizePipeline, test_batch_size, args.threads, args.data, args.dali_cpu)
    speedtest(DecodeFullPipeline, test_batch_size, args.threads, args.data, args.dali_cpu)
    speedtest(DecodeFullPipeline, test_batch_size, args.threads, args.data, args.dali_cpu, True, True)

    print("==="*20)
    speedtest(DecodePipeline, test_batch_size, args.threads, args.data, args.dali_cpu)
    speedtest(DecodeAndCropPipeline, test_batch_size, args.threads, args.data, args.dali_cpu)
    speedtest(DecodeFullSplitPipeline, test_batch_size, args.threads, args.data, args.dali_cpu)
    speedtest(DecodeFullSplitPipeline, test_batch_size, args.threads, args.data, args.dali_cpu, True, True)


    print("==="*20)
    speedtest(DecodeSplitBestPipeline, test_batch_size, args.threads, args.data, args.dali_cpu)

if __name__ == "__main__":
    main()
