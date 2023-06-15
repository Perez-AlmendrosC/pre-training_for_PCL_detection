#!/bin/bash --login

#SBATCH --job-name=rob-base_CV_DPM_bin
#SBATCH --output=rob-base_CV_DPM_bin.out.%J
#SBATCH --error=rob-base_CV_DPM_bin.err.%J
#SBATCH --tasks-per-node=1


#SBATCH --gres=gpu:1
#SBATCH -p gpu             
#SBATCH --ntasks=1
#SBATCH --mem=10000




clush -w $SLURM_NODELIST "sudo /apps/slurm/gpuset_0_shared" 
echo 'Experiment running'

nohup python3 -u clean_binary_dpm!.py --dataset-path='pcl_data' --times-negs=3000 --tokenizer-path='./roberta-base-tokenizer' --model-path='./roberta-base' --adapter-name='cm2' --adapter-path='./adapters/cm/2'

echo 'Finished!'





