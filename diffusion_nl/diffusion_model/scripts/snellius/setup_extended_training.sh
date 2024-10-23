#! /bin/bash

# sbatch ./scripts/snellius/extended_training_agentid.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_agentid_extended_1/DiffusionMultiAgent/aug7wrhb/epoch=94-step=300000.ckpt
# sbatch ./scripts/snellius/extended_training_agentid.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_agentid_extended_2/DiffusionMultiAgent/rhv9r1ee/epoch=94-step=300000.ckpt
# sbatch ./scripts/snellius/extended_training_agentid.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_agentid_extended_3/DiffusionMultiAgent/h0kykmg3/epoch=94-step=300000.ckpt
# sbatch ./scripts/snellius/extended_training_agentid.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_agentid_extended_4/DiffusionMultiAgent/a9szrjvb/epoch=94-step=300000.ckpt

# sbatch ./scripts/snellius/extended_training_action_space.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_agentid_extended_1/DiffusionMultiAgent/bmt7vfe3/epoch=157-step=500000.ckpt
# sbatch ./scripts/snellius/extended_training_action_space.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_agentid_extended_2/DiffusionMultiAgent/tepfz3f4/epoch=157-step=500000.ckpt
# sbatch ./scripts/snellius/extended_training_action_space.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_agentid_extended_3/DiffusionMultiAgent/911pfpq7/epoch=157-step=500000.ckpt
# sbatch ./scripts/snellius/extended_training_action_space.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_agentid_extended_4/DiffusionMultiAgent/cchk7jmj/epoch=157-step=500000.ckpt

sbatch ./scripts/snellius/extended_training_example.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_example_1_1/DiffusionMultiAgent/zu925om7/epoch=157-step=500000.ckpt
sbatch ./scripts/snellius/extended_training_example.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_example_1_2/DiffusionMultiAgent/47hbg1x3/epoch=157-step=500000.ckpt
sbatch ./scripts/snellius/extended_training_example.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_example_1_3/DiffusionMultiAgent/jmb3z1cm/epoch=157-step=500000.ckpt
sbatch ./scripts/snellius/extended_training_example.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_example_1_4/DiffusionMultiAgent/ljr66524/epoch=157-step=500000.ckpt

# sbatch ./scripts/snellius/extended_training_mixed.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_1/DiffusionMultiAgent/npr38vxd/epoch=157-step=500000.ckpt
# sbatch ./scripts/snellius/extended_training_mixed.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_2/DiffusionMultiAgent/ibng0shy/epoch=157-step=500000.ckpt
# sbatch ./scripts/snellius/extended_training_mixed.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_3/DiffusionMultiAgent/r1ypu5yx/epoch=157-step=500000.ckpt
# sbatch ./scripts/snellius/extended_training_mixed.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/mixed_4/DiffusionMultiAgent/qcvihltk/epoch=157-step=500000.ckpt

# sbatch ./scripts/snellius/extended_training_standard.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/standard_5000_edm_1/DiffusionMultiAgent/bhjna6wy/epoch=946-step=500000.ckpt
# sbatch ./scripts/snellius/extended_training_standard.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/standard_5000_edm_2/DiffusionMultiAgent/0uxyrbk6/epoch=946-step=500000.ckpt
# sbatch ./scripts/snellius/extended_training_standard.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/standard_5000_edm_3/DiffusionMultiAgent/1595zr60/epoch=946-step=500000.ckpt
# sbatch ./scripts/snellius/extended_training_standard.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/standard_5000_edm_4/DiffusionMultiAgent/3churtdm/epoch=946-step=500000.ckpt

############## 5000 #############

# #No Left
# # sbatch ./scripts/snellius/extended_training_snellius.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/no_left_5000_edm_1/DiffusionMultiAgent/g8qn925p/epoch=946-step=500000.ckpt ../../data/GOTO/no_left_5000_4_3_False_demos/dataset_5000.pkl no_left_extended
# # sbatch ./scripts/snellius/extended_training_snellius.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/no_left_5000_edm_2/DiffusionMultiAgent/mq0n1r0z/epoch=946-step=500000.ckpt ../../data/GOTO/no_left_5000_4_3_False_demos/dataset_5000.pkl no_left_extended
# # sbatch ./scripts/snellius/extended_training_snellius.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/no_left_5000_edm_3/DiffusionMultiAgent/mldghsbk/epoch=946-step=500000.ckpt ../../data/GOTO/no_left_5000_4_3_False_demos/dataset_5000.pkl no_left_extended
# # sbatch ./scripts/snellius/extended_training_snellius.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/no_left_5000_edm_4/DiffusionMultiAgent/ptwy4viv/epoch=946-step=500000.ckpt ../../data/GOTO/no_left_5000_4_3_False_demos/dataset_5000.pkl no_left_extended

# # No Right 
# sbatch ./scripts/snellius/extended_training_snellius.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/no_right_5000_edm_1/DiffusionMultiAgent/az4zvjth/epoch=946-step=500000.ckpt ../../data/GOTO/no_right_5000_4_3_False_demos/dataset_5000.pkl no_right_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/no_right_5000_edm_2/DiffusionMultiAgent/pztdujy6/epoch=946-step=500000.ckpt ../../data/GOTO/no_right_5000_4_3_False_demos/dataset_5000.pkl no_right_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/no_right_5000_edm_3/DiffusionMultiAgent/qe2mmfyq/epoch=946-step=500000.ckpt ../../data/GOTO/no_right_5000_4_3_False_demos/dataset_5000.pkl no_right_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/no_right_5000_edm_4/DiffusionMultiAgent/8wpv5b7k/epoch=946-step=500000.ckpt ../../data/GOTO/no_right_5000_4_3_False_demos/dataset_5000.pkl no_right_extended

# # Diagonal 
# sbatch ./scripts/snellius/extended_training_snellius.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/diagonal_5000_edm_1/DiffusionMultiAgent/lhx64fav/epoch=946-step=500000.ckpt ../../data/GOTO/diagonal_5000_4_3_False_demos/dataset_5000.pkl diagonal_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/diagonal_5000_edm_2/DiffusionMultiAgent/iv8dmwyp/epoch=946-step=500000.ckpt ../../data/GOTO/diagonal_5000_4_3_False_demos/dataset_5000.pkl diagonal_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/diagonal_5000_edm_3/DiffusionMultiAgent/6hy8wxdu/epoch=946-step=500000.ckpt ../../data/GOTO/diagonal_5000_4_3_False_demos/dataset_5000.pkl diagonal_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/diagonal_5000_edm_4/DiffusionMultiAgent/c44jbcxe/epoch=946-step=500000.ckpt ../../data/GOTO/diagonal_5000_4_3_False_demos/dataset_5000.pkl diagonal_extended

# # WSAD 
# sbatch ./scripts/snellius/extended_training_snellius.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/wsad_5000_edm_1/DiffusionMultiAgent/lmf6l90x/epoch=946-step=500000.ckpt ../../data/GOTO/wsad_5000_4_3_False_demos/dataset_5000.pkl wsad_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/wsad_5000_edm_2/DiffusionMultiAgent/141dzi5j/epoch=946-step=500000.ckpt ../../data/GOTO/wsad_5000_4_3_False_demos/dataset_5000.pkl wsad_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/wsad_5000_edm_3/DiffusionMultiAgent/w5kdgzsa/epoch=946-step=500000.ckpt ../../data/GOTO/wsad_5000_4_3_False_demos/dataset_5000.pkl wsad_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/wsad_5000_edm_4/DiffusionMultiAgent/b0gj0tca/epoch=946-step=500000.ckpt ../../data/GOTO/wsad_5000_4_3_False_demos/dataset_5000.pkl wsad_extended

# # DIR8
# sbatch ./scripts/snellius/extended_training_snellius.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/dir8_5000_edm_1/DiffusionMultiAgent/3e1nd5dy/epoch=946-step=500000.ckpt ../../data/GOTO/dir8_5000_4_3_False_demos/dataset_5000.pkl dir8_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/dir8_5000_edm_2/DiffusionMultiAgent/odxnk5ck/epoch=946-step=500000.ckpt ../../data/GOTO/dir8_5000_4_3_False_demos/dataset_5000.pkl dir8_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/dir8_5000_edm_3/DiffusionMultiAgent/z3wduw2j/epoch=946-step=500000.ckpt ../../data/GOTO/dir8_5000_4_3_False_demos/dataset_5000.pkl dir8_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/dir8_5000_edm_4/DiffusionMultiAgent/rtl0rh5z/epoch=946-step=500000.ckpt ../../data/GOTO/dir8_5000_4_3_False_demos/dataset_5000.pkl dir8_extended

# # Left Right 
# sbatch ./scripts/snellius/extended_training_snellius.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/left_right_5000_edm_1/DiffusionMultiAgent/gevhurn2/epoch=946-step=500000.ckpt  ../../data/GOTO/left_right_5000_4_3_False_demos/dataset_5000.pkl left_right_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/left_right_5000_edm_2/DiffusionMultiAgent/t5x7wi5a/epoch=946-step=500000.ckpt ../../data/GOTO/left_right_5000_4_3_False_demos/dataset_5000.pkl left_right_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/left_right_5000_edm_3/DiffusionMultiAgent/u16g7b31/epoch=946-step=500000.ckpt ../../data/GOTO/left_right_5000_4_3_False_demos/dataset_5000.pkl left_right_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/left_right_5000_edm_4/DiffusionMultiAgent/2efnxoyo/epoch=946-step=500000.ckpt ../../data/GOTO/left_right_5000_4_3_False_demos/dataset_5000.pkl left_right_extended

# # All Diagonal
# sbatch ./scripts/snellius/extended_training_snellius.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/all_diagonal_5000_edm_1/DiffusionMultiAgent/jwd4jhfa/epoch=946-step=500000.ckpt ../../data/GOTO/all_diagonal_5000_4_3_False_demos/dataset_5000.pkl all_diagonal_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/all_diagonal_5000_edm_2/DiffusionMultiAgent/igs7k15s/epoch=946-step=500000.ckpt ../../data/GOTO/all_diagonal_5000_4_3_False_demos/dataset_5000.pkl all_diagonal_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/all_diagonal_5000_edm_3/DiffusionMultiAgent/1ygxi3cq/epoch=946-step=500000.ckpt ../../data/GOTO/all_diagonal_5000_4_3_False_demos/dataset_5000.pkl all_diagonal_extended
# sbatch ./scripts/snellius/extended_training_snellius.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/all_diagonal_5000_edm_4/DiffusionMultiAgent/ujsvcc5z/epoch=946-step=500000.ckpt ../../data/GOTO/all_diagonal_5000_4_3_False_demos/dataset_5000.pkl all_diagonal_extended

############## 30000 #############

# Standard 
# sbatch ./scripts/snellius/extended_training_snellius.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/standard_30000_edm_1/DiffusionMultiAgent/2y4slui1/epoch=157-step=500000.ckpt ../../data/GOTO/standard_30000_4_3_False_demos/dataset_30000.pkl standard_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/standard_30000_edm_2/DiffusionMultiAgent/4d1g38fk/epoch=157-step=500000.ckpt ../../data/GOTO/standard_30000_4_3_False_demos/dataset_30000.pkl standard_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/standard_30000_edm_3/DiffusionMultiAgent/6rmfsw3g/epoch=157-step=500000.ckpt ../../data/GOTO/standard_30000_4_3_False_demos/dataset_30000.pkl standard_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/standard_30000_edm_4/DiffusionMultiAgent/0kusaeug/epoch=157-step=500000.ckpt ../../data/GOTO/standard_30000_4_3_False_demos/dataset_30000.pkl standard_extended_30000

# # No Left
# sbatch ./scripts/snellius/extended_training_snellius.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/no_left_30000_edm_1/DiffusionMultiAgent/cjpyq9s1/epoch=157-step=500000.ckpt ../../data/GOTO/no_left_30000_4_3_False_demos/dataset_30000.pkl no_left_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/no_left_30000_edm_2/DiffusionMultiAgent/vwm740f2/epoch=157-step=500000.ckpt ../../data/GOTO/no_left_30000_4_3_False_demos/dataset_30000.pkl no_left_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/no_left_30000_edm_3/DiffusionMultiAgent/yjwoesxe/epoch=157-step=500000.ckpt ../../data/GOTO/no_left_30000_4_3_False_demos/dataset_30000.pkl no_left_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/no_left_30000_edm_4/DiffusionMultiAgent/0yg9h2z8/epoch=157-step=500000.ckpt ../../data/GOTO/no_left_30000_4_3_False_demos/dataset_30000.pkl no_left_extended_30000

# # No Right 
# sbatch ./scripts/snellius/extended_training_snellius.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/no_right_30000_edm_1/DiffusionMultiAgent/gr1noqfb/epoch=157-step=500000.ckpt ../../data/GOTO/no_right_30000_4_3_False_demos/dataset_30000.pkl no_right_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/no_right_30000_edm_2/DiffusionMultiAgent/rtitf4e8/epoch=157-step=500000.ckpt ../../data/GOTO/no_right_30000_4_3_False_demos/dataset_30000.pkl no_right_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/no_right_30000_edm_3/DiffusionMultiAgent/e5vb9sxh/epoch=157-step=500000.ckpt ../../data/GOTO/no_right_30000_4_3_False_demos/dataset_30000.pkl no_right_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/no_right_30000_edm_4/DiffusionMultiAgent/2zmgrjgl/epoch=157-step=500000.ckpt ../../data/GOTO/no_right_30000_4_3_False_demos/dataset_30000.pkl no_right_extended_30000

# # Diagonal 
# sbatch ./scripts/snellius/extended_training_snellius.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/diagonal_30000_edm_1/DiffusionMultiAgent/a2d3146k/epoch=157-step=500000.ckpt ../../data/GOTO/diagonal_30000_4_3_False_demos/dataset_30000.pkl diagonal_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/diagonal_30000_edm_2/DiffusionMultiAgent/47tiey34/epoch=157-step=500000.ckpt ../../data/GOTO/diagonal_30000_4_3_False_demos/dataset_30000.pkl diagonal_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/diagonal_30000_edm_3/DiffusionMultiAgent/i4pkg6yk/epoch=157-step=500000.ckpt ../../data/GOTO/diagonal_30000_4_3_False_demos/dataset_30000.pkl diagonal_extended_30000
#sbatch ./scripts/snellius/extended_training_snellius.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/diagonal_30000_edm_4/DiffusionMultiAgent/s9yxeeaz/epoch=157-step=500000.ckpt ../../data/GOTO/diagonal_30000_4_3_False_demos/dataset_30000.pkl diagonal_extended_30000

# # WSAD 
# sbatch ./scripts/snellius/extended_training_snellius.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/wsad_30000_edm_1/DiffusionMultiAgent/0beu20hy/epoch=157-step=500000.ckpt ../../data/GOTO/wsad_30000_4_3_False_demos/dataset_30000.pkl wsad_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/wsad_30000_edm_2/DiffusionMultiAgent/wublu62d/epoch=157-step=500000.ckpt ../../data/GOTO/wsad_30000_4_3_False_demos/dataset_30000.pkl wsad_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/wsad_30000_edm_3/DiffusionMultiAgent/ypfz3ero/epoch=157-step=500000.ckpt ../../data/GOTO/wsad_30000_4_3_False_demos/dataset_30000.pkl wsad_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/wsad_30000_edm_4/DiffusionMultiAgent/txlctm2l/epoch=157-step=500000.ckpt ../../data/GOTO/wsad_30000_4_3_False_demos/dataset_30000.pkl wsad_extended_30000

# # DIR8
# sbatch ./scripts/snellius/extended_training_snellius.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/dir8_30000_edm_1/DiffusionMultiAgent/z7llogla/epoch=157-step=500000.ckpt ../../data/GOTO/dir8_30000_4_3_False_demos/dataset_30000.pkl dir8_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/dir8_30000_edm_2/DiffusionMultiAgent/03ii4xpq/epoch=157-step=500000.ckpt ../../data/GOTO/dir8_30000_4_3_False_demos/dataset_30000.pkl dir8_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/dir8_30000_edm_3/DiffusionMultiAgent/q9hh2y9c/epoch=157-step=500000.ckpt ../../data/GOTO/dir8_30000_4_3_False_demos/dataset_30000.pkl dir8_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/dir8_30000_edm_4/DiffusionMultiAgent/rca45vgb/epoch=157-step=500000.ckpt ../../data/GOTO/dir8_30000_4_3_False_demos/dataset_30000.pkl dir8_extended_30000

# # Left Right 
# sbatch ./scripts/snellius/extended_training_snellius.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/left_right_30000_edm_1/DiffusionMultiAgent/nq5t3xaj/epoch=157-step=500000.ckpt ../../data/GOTO/left_right_30000_4_3_False_demos/dataset_30000.pkl left_right_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/left_right_30000_edm_2/DiffusionMultiAgent/j9zc1xg5/epoch=157-step=500000.ckpt ../../data/GOTO/left_right_30000_4_3_False_demos/dataset_30000.pkl left_right_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/left_right_30000_edm_3/DiffusionMultiAgent/6f758ud8/epoch=157-step=500000.ckpt ../../data/GOTO/left_right_30000_4_3_False_demos/dataset_30000.pkl left_right_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/left_right_30000_edm_4/DiffusionMultiAgent/i035l71v/epoch=157-step=500000.ckpt ../../data/GOTO/left_right_30000_4_3_False_demos/dataset_30000.pkl left_right_extended_30000

# # All Diagonal
# sbatch ./scripts/snellius/extended_training_snellius.sh 1 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/all_diagonal_30000_edm_1/DiffusionMultiAgent/9lhzc0jf/epoch=157-step=500000.ckpt ../../data/GOTO/all_diagonal_30000_4_3_False_demos/dataset_30000.pkl all_diagonal_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 2 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/all_diagonal_30000_edm_2/DiffusionMultiAgent/99edeqqf/epoch=157-step=500000.ckpt ../../data/GOTO/all_diagonal_30000_4_3_False_demos/dataset_30000.pkl all_diagonal_extended_30000
# sbatch ./scripts/snellius/extended_training_snellius.sh 3 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/all_diagonal_30000_edm_3/DiffusionMultiAgent/1s69myi7/epoch=157-step=500000.ckpt ../../data/GOTO/all_diagonal_30000_4_3_False_demos/dataset_30000.pkl all_diagonal_extended_30000
#sbatch ./scripts/snellius/extended_training_snellius.sh 4 /scratch-shared/nhoepner/experiments/diffusion_nl2/diffusion_nl/model_store/all_diagonal_30000_edm_4/DiffusionMultiAgent/zukkxf4i/epoch=157-step=500000.ckpt ../../data/GOTO/all_diagonal_30000_4_3_False_demos/dataset_30000.pkl all_diagonal_extended_30000