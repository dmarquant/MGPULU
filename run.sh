#!/bin/bash

for i in $(seq 1 5)
do
    CUDA_VISIBLE_DEVICES=0 ./single_lu $(( $i * 8192 ))
    sleep 10s

    CUDA_VISIBLE_DEVICES=0,1 ./ferry_lu $(( $i * 8192 ))
    sleep 10s
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./ferry_lu $(( $i * 8192 ))
    sleep 10s

    CUDA_VISIBLE_DEVICES=0,1 ./lu $(( $i * 8192 ))
    sleep 10s
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./lu $(( $i * 8192 ))
    sleep 10s

    CUDA_VISIBLE_DEVICES=0,1 ./task_lu $(( $i * 8192 ))
    sleep 10s
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./task_lu $(( $i * 8192 ))
    sleep 10s
done
