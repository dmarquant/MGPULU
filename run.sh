#!/bin/bash


for i in $(seq 1 6)
do
    CUDA_VISIBLE_DEVICES=0 ./lu $(( $i * 8192 ))
    sleep 2m

    CUDA_VISIBLE_DEVICES=0,1 ./lu $(( $i * 8192 ))
    sleep 2m

    CUDA_VISIBLE_DEVICES=0,1,2,3 ./lu $(( $i * 8192 ))
    sleep 2m
done
