#!/bin/bash --login
#SBATCH --job-name=hate2ep_lr=1e-4
#SBATCH --output=hate2ep_lr=1e-4.out.%J
#SBATCH --error=hate2ep_lr=1e-4.err.%J
#SBATCH --tasks-per-node=7


#SBATCH --gres=gpu:1
#SBATCH -p gpu
##SBATCH -p dev ##for dev partition

#SBATCH --ntasks=7

clush -w $SLURM_NODELIST "sudo /apps/slurm/gpuset_0_shared" 

echo 'Pretraining adapter'

nohup python3 -u pretrain_adapter.py --dataset-name='hate' --dataset-path='clean_datasets/hate_clean_nonans.csv' --output-path='adapters/hate/2' --model-name='roBERTa' --model-path='./roberta-for-sequence-classification' --num-epochs=2

echo 'Finished!'





