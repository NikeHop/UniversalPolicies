#! /bin/bash

set -e 

# Standard
# python ./eval.py --config ./configs/standard/distractors.yaml --dm_model_path "standard_extended_30000_1/DiffusionMultiAgent" --dm_model_name frkamt7l --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_standard_distractor_30000_1  --device "cuda:0" 

# python ./eval.py --config ./configs/standard/distractors.yaml --dm_model_path "standard_extended_30000_2/DiffusionMultiAgent" --dm_model_name xdto8eya --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_standard_distractor_30000_2  --device "cuda:0" 

# python ./eval.py --config ./configs/standard/distractors.yaml --dm_model_path "standard_extended_30000_3/DiffusionMultiAgent" --dm_model_name h8tyfswk --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_standard_distractor_30000_3  --device "cuda:0"

# python ./eval.py --config ./configs/standard/distractors.yaml --dm_model_path "standard_extended_30000_4/DiffusionMultiAgent" --dm_model_name xg45eccm --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_standard_distractor_30000_4  --device "cuda:0" 

# # No left
# python ./eval.py --config ./configs/no_left/distractors.yaml --dm_model_path "no_left_extended_30000_1/DiffusionMultiAgent" --dm_model_name ycszunaj --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_no_left_distractor_30000_1  --device "cuda:0" 

# python ./eval.py --config ./configs/no_left/distractors.yaml --dm_model_path "no_left_extended_30000_2/DiffusionMultiAgent" --dm_model_name klzhwlqm --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_no_left_distractor_30000_2  --device "cuda:0" 

# python ./eval.py --config ./configs/no_left/distractors.yaml --dm_model_path "no_left_extended_30000_3/DiffusionMultiAgent" --dm_model_name n2y2d72m --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_no_left_distractor_30000_3  --device "cuda:0"

# python ./eval.py --config ./configs/no_left/distractors.yaml --dm_model_path "no_left_extended_30000_4/DiffusionMultiAgent" --dm_model_name mflmmchb --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_no_left_distractor_30000_4  --device "cuda:0" 

# # No right 
# python ./eval.py --config ./configs/no_right/distractors.yaml --dm_model_path "no_right_extended_30000_1/DiffusionMultiAgent" --dm_model_name p4mp8ro5 --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_no_right_distractor_30000_1  --device "cuda:0" 

# python ./eval.py --config ./configs/no_right/distractors.yaml --dm_model_path "no_right_extended_30000_2/DiffusionMultiAgent" --dm_model_name dkrkkyjv --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_no_right_distractor_30000_2  --device "cuda:0" 

# python ./eval.py --config ./configs/no_right/distractors.yaml --dm_model_path "no_right_extended_30000_3/DiffusionMultiAgent" --dm_model_name hz03svg1 --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_no_right_distractor_30000_3  --device "cuda:0"

# python ./eval.py --config ./configs/no_right/distractors.yaml --dm_model_path "no_right_extended_30000_4/DiffusionMultiAgent" --dm_model_name fjab5793 --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_no_right_distractor_30000_4  --device "cuda:0" 

# # Diagonal
# python ./eval.py --config ./configs/diagonal/distractors.yaml --dm_model_path "diagonal_extended_30000_1/DiffusionMultiAgent" --dm_model_name 7q0dh67a --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_daigonal_distractor_30000_1  --device "cuda:0" 

# python ./eval.py --config ./configs/diagonal/distractors.yaml --dm_model_path "diagonal_extended_30000_2/DiffusionMultiAgent" --dm_model_name ud3drj7e --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_daigonal_distractor_30000_2  --device "cuda:0" 

# python ./eval.py --config ./configs/diagonal/distractors.yaml --dm_model_path "diagonal_extended_30000_3/DiffusionMultiAgent" --dm_model_name 0g06huzm --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_daigonal_distractor_30000_3  --device "cuda:0"

# python ./eval.py --config ./configs/diagonal/distractors.yaml --dm_model_path "diagonal_extended_30000_4/DiffusionMultiAgent" --dm_model_name 26geen9v --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_daigonal_distractor_30000_4  --device "cuda:0" 

# # WSAD
python ./eval.py --config ./configs/wsad/distractors.yaml --dm_model_path "wsad_extended_30000_1/DiffusionMultiAgent" --dm_model_name hxrzovna --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_wsad_distractor_30000_1  --device "cuda:0" 

python ./eval.py --config ./configs/wsad/distractors.yaml --dm_model_path "wsad_extended_30000_2/DiffusionMultiAgent" --dm_model_name gr9p9ju9 --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_wsad_distractor_30000_2  --device "cuda:0" 

python ./eval.py --config ./configs/wsad/distractors.yaml --dm_model_path "wsad_extended_30000_3/DiffusionMultiAgent" --dm_model_name 5ho7kuq8 --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_wsad_distractor_30000_3  --device "cuda:0"

python ./eval.py --config ./configs/wsad/distractors.yaml --dm_model_path "wsad_extended_30000_4/DiffusionMultiAgent" --dm_model_name thjb2hb8 --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_wsad_distractor_30000_4  --device "cuda:0" 

# # Dir8
python ./eval.py --config ./configs/dir8/distractors.yaml --dm_model_path "dir8_extended_30000_1/DiffusionMultiAgent" --dm_model_name dd9l2f5a  --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_dir8_distractor_30000_1  --device "cuda:0" 

python ./eval.py --config ./configs/dir8/distractors.yaml --dm_model_path "dir8_extended_30000_2/DiffusionMultiAgent" --dm_model_name kd6f436w --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_dir8_distractor_30000_2  --device "cuda:0" 

python ./eval.py --config ./configs/dir8/distractors.yaml --dm_model_path "dir8_extended_30000_3/DiffusionMultiAgent" --dm_model_name wsk82rd0 --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_dir8_distractor_30000_3  --device "cuda:0"

python ./eval.py --config ./configs/dir8/distractors.yaml --dm_model_path "dir8_extended_30000_4/DiffusionMultiAgent" --dm_model_name k7vkrtlt --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_dir8_distractor_30000_4  --device "cuda:0" 

# # Left right
python ./eval.py --config ./configs/left_right/distractors.yaml --dm_model_path "left_right_extended_30000_1/DiffusionMultiAgent" --dm_model_name 09rl833d --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_left_right_distractor_30000_1  --device "cuda:0" 

python ./eval.py --config ./configs/left_right/distractors.yaml --dm_model_path "left_right_extended_30000_2/DiffusionMultiAgent" --dm_model_name 0g4n9eyf --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_left_right_distractor_30000_2  --device "cuda:0" 

python ./eval.py --config ./configs/left_right/distractors.yaml --dm_model_path "left_right_extended_30000_3/DiffusionMultiAgent" --dm_model_name qc5yzhn2 --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_left_right_distractor_30000_3  --device "cuda:0"

python ./eval.py --config ./configs/left_right/distractors.yaml --dm_model_path "left_right_extended_30000_4/DiffusionMultiAgent" --dm_model_name z1g04br1 --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_left_right_distractor_30000_4  --device "cuda:0" 

# # All diagonal
python ./eval.py --config ./configs/all_diagonal/distractors.yaml --dm_model_path "all_diagonal_extended_30000_1/DiffusionMultiAgent" --dm_model_name iej2e98h --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_all_diagonal_distractor_30000_1  --device "cuda:0" 

python ./eval.py --config ./configs/all_diagonal/distractors.yaml --dm_model_path "all_diagonal_extended_30000_2/DiffusionMultiAgent" --dm_model_name ezdy8u95 --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_all_diagonal_distractor_30000_2  --device "cuda:0" 

python ./eval.py --config ./configs/all_diagonal/distractors.yaml --dm_model_path "all_diagonal_extended_30000_3/DiffusionMultiAgent" --dm_model_name ary5eptn --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_all_diagonal_distractor_30000_3  --device "cuda:0"

python ./eval.py --config ./configs/all_diagonal/distractors.yaml --dm_model_path "all_diagonal_extended_30000_4/DiffusionMultiAgent" --dm_model_name whp93z9r  --dm_model_checkpoint "epoch=157-step=500000.ckpt" --experiment_name eval_all_diagonal_distractor_30000_4  --device "cuda:0" 

