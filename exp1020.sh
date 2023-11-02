#!/bin/bash

# > mia_result1019.txt
# 定义多组 coreset 和 defense 参数
# "ContextualDiversity" "Craig" "DeepFool" "Forgetting" "Glister" "Herding"     "Cal" "GraNd" "Uncertainty"
# "dpsgd"    "early_stopping"     "vanilla" "advreg" "confidence_penalty" "distillation" "distillation" "label_smoothing" 
model_options=("resnet18")
loss_options=("Focal")
lr_options=(1e-1)
gamma_options=(0.5 0.75 1.0 1.25 1.5 1.75 2 2.25 2.5 2.75 3)

# 循环运行脚本
for model in "${model_options[@]}"; do
    for loss in "${loss_options[@]}"; do
        for lr in "${lr_options[@]}"; do
            for gamma in "${gamma_options[@]}"; do
                python Assignment01.py --lr="$lr" --loss="$loss" --model="$model" --gamma="$gamma"
            done
        done
    done
done