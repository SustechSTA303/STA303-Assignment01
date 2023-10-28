#!/bin/sh

MODEL_LIST=(baseline resnet18)
GAMMA_LIST=(0 0.25 0.5 0.75 1 1.25 1.5 1.75 2)

for MODEL in "${MODEL_LIST[@]}"
    do
    for GAMMA in "${GAMMA_LIST[@]}"
    do
        echo ==========================
        echo Model: $MODEL, FOCAL GAMMA: $GAMMA
        echo focal_eval.py --model $MODEL --gamma $GAMMA
        echo ==========================
        python focal_eval.py --model $MODEL --gamma $GAMMA
    done
done
