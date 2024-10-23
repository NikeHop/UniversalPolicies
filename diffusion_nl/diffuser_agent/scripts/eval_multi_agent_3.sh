#! /bin/bash


set -e 


# python ./eval.py --config ./configs/standard/distractors.yaml --dm_model_path "standard 30000/DiffusionMultiAgent" --dm_model_name "rz7onwxe" --dm_model_checkpoint "epoch=252-step=800000.ckpt"  --device "cuda:0" &
# pid1=$!
# sleep 5

# python ./eval.py --config ./configs/no_left/distractors.yaml --dm_model_path "no left 30000/DiffusionMultiAgent" --dm_model_name "5kbg87aj" --dm_model_checkpoint "epoch=252-step=800000.ckpt"  --device "cuda:1"  &
# pid2=$!
# sleep 5

# python ./eval.py --config ./configs/no_right/distractors.yaml --dm_model_path "no right 30000/DiffusionMultiAgent" --dm_model_name "ytt6f9u6" --dm_model_checkpoint "epoch=252-step=800000.ckpt"  --device "cuda:2" &
# pid3=$!
# sleep 5

# python ./eval.py --config ./configs/diagonal/distractors.yaml --dm_model_path "diagonal 30000/DiffusionMultiAgent" --dm_model_name "1pyq29xh" --dm_model_checkpoint "epoch=252-step=800000.ckpt"  --device "cuda:3"  &
# pid4=$!
# sleep 5

# wait $pid1 $pid2 $pid3 $pid4

python ./eval.py --config ./configs/dir8/distractors.yaml --dm_model_path "dir8 30000/DiffusionMultiAgent" --dm_model_name "le3dpwou" --dm_model_checkpoint "epoch=252-step=800000.ckpt"  --device "cuda:0" &
pid1=$!
sleep 5 

wait $pid1
# python ./eval.py --config ./configs/wsad/distractors.yaml --dm_model_path "wsad 30000/DiffusionMultiAgent" --dm_model_name "sonurhdc" --dm_model_checkpoint "epoch=252-step=800000.ckpt"  --device "cuda:1" &
# pid2=$!
# sleep 5

# python ./eval.py --config ./configs/left_right/distractors.yaml --dm_model_path "left_right 30000/DiffusionMultiAgent" --dm_model_name "xrb2r9wn" --dm_model_checkpoint "epoch=252-step=800000.ckpt"  --device "cuda:2" &
# pid3=$!
# sleep 5

# python ./eval.py --config ./configs/all_diagonal/distractors.yaml --dm_model_path "all_diagonal 30000/DiffusionMultiAgent" --dm_model_name "cx1o6nqy" --dm_model_checkpoint "epoch=252-step=800000.ckpt"  --device "cuda:3"  &
# pid4=$!
# sleep 5

# wait $pid1 $pid2 $pid3 $pid4
