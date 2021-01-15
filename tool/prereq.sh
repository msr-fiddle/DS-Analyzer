#!/bin/bash
g++ memory_thr.cc -fopenmp -lrt -o memtest
pip install tqdm
pip install colorama
