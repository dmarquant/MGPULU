#!/bin/bash


for i in $(seq 1 6)
do
    #CUDA_VISIBLE_DEVICES=0 ./lu $(( $i * 8192 ))
    #sleep 2m

    CUDA_VISIBLE_DEVICES=0,1 ./task_lu $(( $i * 8192 ))
    sleep 1m

    CUDA_VISIBLE_DEVICES=0,1,2,3 ./task_lu $(( $i * 8192 ))
    sleep 1m
done
