#!/bin/bash

: '
This code evaluates the performance of the model for all epochs.
Then it runs the code that find the best validation epoch and uses it to calculate the performance of the model.

To run the code for interaction prediction on the reddit dataset: 
$ ./evaluate_all_epochs.sh reddit interaction

To run the code for state change prediction on the reddit dataset: 
$ ./evaluate_all_epochs.sh reddit state
'

network=$1
type=$2
gpu=$3
alpha=$4
interaction="interaction"

idx=70
while [ $idx -le 99 ]
do
    echo $idx
    if [ $type == "$interaction" ]; then
	python3 evaluate_interaction_prediction.py --network $network --model DGCF --epoch ${idx} --method attention --adj --alpha $alpha --gpu $gpu
    else
	python3 evaluate_state_change_prediction.py --network $network --model DGCF --epoch ${idx} --method attention --adj --alpha $alpha --gpu $gpu
    fi
    (( idx+=1 ))
done 


if [ $type == "$interaction" ]; then
    python3 get_final_performance_numbers.py results/interaction_prediction_${network}.attention.adj.txt --alpha $alpha --gpu $gpu
else
    python3 get_final_performance_numbers.py results/state_change_prediction_${network}.txt --alpha $alpha --gpu $gpu
fi
