#! /bin/bash


set -e

for i in 0 1 2 3 4 5 6 7
do
python eval.py --config ./configs/eval_instruction_imitation_goto.yaml --checkpoint $1 --action_space $i
done
