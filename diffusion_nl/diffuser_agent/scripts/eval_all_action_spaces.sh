#! /bin/bash 

set -e 

for i in 0 1 2 3 4 5 6 7
do
python eval.py --config ./configs/goto.yaml --action_space $i --checkpoint $1
done