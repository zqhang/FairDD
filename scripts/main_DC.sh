#!/bin/bash

model=('ConvNet')

dataset=Colored_MNIST_foreground
# dataset=Colored_FashionMNIST_foreground
# dataset=Colored_MNIST_background
# dataset=Colored_FashionMNIST_background
# dataset=CIFAR10_S_90
# dataset=CelebA
# dataset=UTKface
# dataset=BFFHQ

gpu_id=1
num_exp=1
num_eval=10
ipc=10
batch_real=256
note=DC_logs

for i in "${!model[@]}";do

  for dataset in "${dataset_list[@]}"; do

    name="DC_ipc${ipc}"

    root_dir='.'
    save_path="${root_dir}/logs/${note}/${dataset}/${model}/${name}/"
    LOG="${save_path}res.log"
    mkdir -p "${root_dir}/logs/${note}/${dataset}/${model}/${name}"

    {
    echo Logging output to "$LOG"
    echo ${dataset}
    echo ${ipc}

    CUDA_VISIBLE_DEVICES=${gpu_id} python3 main_DC.py \
     --dataset ${dataset}\
     --ipc ${ipc}\
     --batch_real ${batch_real}\
     --num_exp ${num_exp}\
     --model ${model}\
     --save_path ${save_path}\
     --num_eval ${num_eval}\

    } &> >(tee -a "$LOG")

  done
done


for i in "${!model[@]}";do
  for dataset in "${dataset_list[@]}"; do

    name="DC_ipc${ipc}_FairDD"

    root_dir='.'
    save_path="${root_dir}/logs/${note}/${dataset}/${model}/${name}/"
    LOG="${save_path}res.log"
    mkdir -p "${root_dir}/logs/${note}/${dataset}/${model}/${name}"

    {
    echo Logging output to "$LOG"
    echo ${dataset}
    echo ${ipc}

    CUDA_VISIBLE_DEVICES=${gpu_id} python3 main_DC.py \
     --dataset ${dataset}\
     --ipc ${ipc}\
     --batch_real ${batch_real}\
     --num_exp ${num_exp}\
     --model ${model}\
     --save_path ${save_path}\
     --num_eval ${num_eval}\
     --FairDD \

    } &> >(tee -a "$LOG")

  done
done
