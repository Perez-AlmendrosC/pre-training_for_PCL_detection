#!/bin/bash --login
#SBATCH -A scw1787
#SBATCH --job-name=ft_stereo2_3k_set4_binary
#SBATCH --output=ft_stereo2_3k_set4_binary.out.%J
#SBATCH --error=ft_stereo2_3k_set4_binary.err.%J
#SBATCH --tasks-per-node=7



#SBATCH --gres=gpu:1
#SBATCH -p gpu               
#SBATCH --ntasks=1



cd ~/adapters_env
module load anaconda/2020.02
source activate
conda activate adapters_env

clush -w $SLURM_NODELIST "sudo /apps/slurm/gpuset_0_shared" #"sudo /apps/slurm/gpuset_0_shared"  #"sudo /apps/slurm/gpuset_3_exclusive" 
echo 'Experiment running'

echo 'roBERTa + adapters + pcl'


#nohup python3 -u setting4_multilabel+adapters.py --dataset-path='pcl_data' --times-negs=3000 --tokenizer-path='/scratch/c.c1867383/roberta-tokenizer' --model-path='/scratch/c.c1867383/roberta-for-sequence-classification' --adapter-name='offen2' --adapter-path='/scratch/c.c1867383/adapters/offensive/2' --results-path='/results'
#nohup python3 -u setting4_binary_+adapters.py --dataset-path='pcl_data' --times-negs=3000 --tokenizer-path='/scratch/c.c1867383/roberta-tokenizer' --model-path='/scratch/c.c1867383/roberta-for-sequence-classification_2' --adapter-name='cm5' --adapter-path='/scratch/c.c1867383/adapters/cm/5' --results-path='/results'
nohup python3 -u setting4_binary_+ftmodel.py --dataset-path='pcl_data' --times-negs=3000 --tokenizer-path='/scratch/c.c1867383/roberta-tokenizer' --model-path='/scratch/c.c1867383/models/stereo_2' --model-name='stereoset_2' --results-path='/results'


echo 'Code running'

echo 'Finished!'

###SBATCH --mem=256


