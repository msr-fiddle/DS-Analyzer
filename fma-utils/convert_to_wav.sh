#!/bin/bash
if [ "$#" -ne 2 ]; then 
   echo "Usage ./convert.. <input dir(train/val) cotaining file list> <out dir>"
   exit 1
fi

apt-get install ffmpeg

input=$1
out_dir=$2

for d in $input/*; do
    dir_name=$d
    working_dir=$out_dir/$dir_name
    mkdir -p $working_dir   
    echo "Input $dir_name, out $working_dir" 
    ./_convert.sh $dir_name $out_dir & pids+=($!)  
done
wait "${pids[@]}"
echo "Completed"
