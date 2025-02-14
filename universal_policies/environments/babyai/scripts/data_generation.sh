#! /bin/bash

set -e 

python generate_demos.py --config $1 --action_space 0
python generate_demos.py --config $1 --action_space 1
python generate_demos.py --config $1 --action_space 2
python generate_demos.py --config $1 --action_space 3
python generate_demos.py --config $1 --action_space 4
python generate_demos.py --config $1 --action_space 5
python generate_demos.py --config $1 --action_space 6
python generate_demos.py --config $1 --action_space 7

python merge.py --config $1