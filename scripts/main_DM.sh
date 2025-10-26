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

gpu_id=0
num_exp=1
num_eval=10
ipc=10
batch_real=256
Iteration=20000
note=DM_logs

for i in "${!model[@]}"; do
  for dataset in "${dataset_list[@]}"; do

    name="DM_ipc${ipc}"

    # Check if ipc is 100, then set lr_img to 10, otherwise set to 1
    if [ $ipc -eq 100 ]; then
      lr_img=10
    else
      lr_img=1
    fi

    root_dir='.'
    save_path="${root_dir}/logs/${note}/${dataset}/${model}/${name}/"
    LOG="${save_path}res.log"
    mkdir -p "${root_dir}/logs/${note}/${dataset}/${model}/${name}"

    {
    echo Logging output to "$LOG"
    echo ${dataset}
    echo ${ipc}

    CUDA_VISIBLE_DEVICES=${gpu_id} python3 main_DM.py \
      --dataset ${dataset}\
      --ipc ${ipc}\
      --batch_real ${batch_real}\
      --num_exp ${num_exp}\
      --model ${model}\
      --save_path ${save_path}\
      --num_eval ${num_eval}\
      --lr_img ${lr_img}\
      --Iteration ${Iteration}\

    } &> >(tee -a "$LOG")

  done
done



for i in "${!model[@]}"; do
  for dataset in "${dataset_list[@]}"; do

    name="DM_ipc${ipc}_FairDD"

    # Check if ipc is 100, then set lr_img to 10, otherwise set to 1
    if [ $ipc -eq 100 ]; then
      lr_img=10
    else
      lr_img=1
    fi

    root_dir='.'
    save_path="${root_dir}/logs/${note}/${dataset}/${model}/${name}/"
    LOG="${save_path}res.log"
    mkdir -p "${root_dir}/logs/${note}/${dataset}/${model}/${name}"

    {
    echo Logging output to "$LOG"
    echo ${dataset}
    echo ${ipc}

    CUDA_VISIBLE_DEVICES=${gpu_id} python3 main_DM.py \
      --dataset ${dataset}\
      --ipc ${ipc}\
      --batch_real ${batch_real}\
      --num_exp ${num_exp}\
      --model ${model}\
      --save_path ${save_path}\
      --num_eval ${num_eval}\
      --lr_img ${lr_img}\
      --Iteration ${Iteration}\
      --FairDD \

    } &> >(tee -a "$LOG")

  done
done

