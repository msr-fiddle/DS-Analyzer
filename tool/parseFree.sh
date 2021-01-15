#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Please input the free log"
	exit 1
fi

log=$1
log_tmp='tmp.txt'
csv="${log%.*}".csv
sed '/^Swap/ d' < $log  > $log_tmp
sed '/total/ d' < $log_tmp > $log

echo "None,total,used,free,shared,cache,avail" > $csv
tr -s ' ' ',' < $log >> $csv
