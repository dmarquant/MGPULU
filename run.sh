#!/bin/bash


for i in $(seq 1 4)
do
    CUDA_VISIBLE_DEVICES=0 ./lu $(( $i * 8192 ))
    CUDA_VISIBLE_DEVICES=0,1 ./lu $(( $i * 8192 ))
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./lu $(( $i * 8192 ))
done
