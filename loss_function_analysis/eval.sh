#!/bin/sh

MODEL_LIST=(baseline resnet18)
LOSS_LIST=(mae ce focal0.5 focal2)

for MODEL in "${MODEL_LIST[@]}"
do
    for LOSS in "${LOSS_LIST[@]}"
    do
        echo ==========================
        echo Model: $MODEL, Loss: $LOSS
        echo CIFAR10_eval.py --model $MODEL --loss $LOSS
        echo ==========================
        python CIFAR!0_eval.py --model $MODEL --loss $LOSS
    done
done

