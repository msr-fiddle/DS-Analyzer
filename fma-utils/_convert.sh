#!/bin/bash
dir_name=$1
out_dir=$2


for file in $dir_name/*; do  
    infile=$file 
    filename="${file%.*}.wav"  
    outfile=$out_dir/$filename    
    tmpfile=$out_dir/$dir_name/tmp.wav
    echo "infile=$infile, outfile=$outfile, tmpfile=$tmpfile"
    ffmpeg -loglevel panic -i $infile $tmpfile    
    duration=`ffprobe -i $tmpfile -show_entries format=duration -v  quiet -of csv="p=0"` 
    duration=${duration%.*}   
    #echo "Duration is $duration"   
    target=$((duration/5))    
    echo "Copying $target seconds of $tmpfile to $outfile"     
    ffmpeg -loglevel panic -i $tmpfile -ss 0 -to $target -c copy $outfile   
    rm $tmpfile 
done
