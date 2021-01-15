
Data Stall Profiler

Usage:
./prereq.sh


python harness.py --nproc_per_node=8 -j 3 -b 512  -a alexnet --prefix results/gpu-prep/  --classes 1000  image_classification/pytorch-imagenet-dali-mp.py --amp --data-profile --noeval --full_epoch /datadrive/mnt2/jaya/datasets/imagenet/


python harness.py --nproc_per_node=8 -j 3 -b 512  -a alexnet --resume_json alexnet/gpus-8/run1-synthetic/MODEL.json image_classification/pytorch-imagenet-dali-mp.py --amp --data-profile /datadrive/mnt2/jaya/datasets/imagenet/

If running for the first time, donot pass in resume args. 
The final MODEL.json file containing perf results can be found at
        <tool_dir>/<model>/<num_jobs>/<num_gpu>/<num_cpu>/MODEL.json

python what_if_tool.py --path <path_to_model> 


