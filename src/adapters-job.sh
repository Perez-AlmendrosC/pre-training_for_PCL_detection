#!/bin/bash --login
#SBATCH -A scw1787
#SBATCH --job-name=hate2_heads_rec_3k_set4_binary
#SBATCH --output=hate2_heads_rec_3k_set4_binary.out.%J
#SBATCH --error=hate2_heads_rec_3k_set4_binary.err.%J
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

nohup python3 -u setting4_binary_+adapters.py --dataset-path='pcl_data' --times-negs=3000 --tokenizer-path='/scratch/c.c1867383/roberta-tokenizer' --model-path='/scratch/c.c1867383/roberta-for-sequence-classification_2' --adapter-name='hate2_h' --adapter-path='/scratch/c.c1867383/adapters/hate/withheads_2' --results-path='/results'
#nohup python3 -u setting4_binary_+ftmodel.py --dataset-path='pcl_data' --times-negs=3000 --tokenizer-path='/scratch/c.c1867383/roberta-tokenizer' --model-path='/scratch/c.c1867383/models/justice_2' --model-name='justice_2' --results-path='/results'

#nohup python3 -u setting4_multilabel+adapters.py --dataset-path='pcl_data' --times-negs=3000 --tokenizer-path='/scratch/c.c1867383/roberta-tokenizer' --model-path='/scratch/c.c1867383/roberta-for-sequence-classification' --adapter-name='cm2' --adapter-path='/scratch/c.c1867383/adapters/cm/2' --results-path='/results'
#nohup python3 -u setting4_binary_+adapters.py --dataset-path='pcl_data' --times-negs=3000 --tokenizer-path='/scratch/c.c1867383/roberta-tokenizer' --model-path='/scratch/c.c1867383/roberta-for-sequence-classification_2' --results-path='/results'


#python3 baseline_wdownsampling.py --dataset-path='pcl_data' --times-negs=3000 --tokenizer-path='/scratch/c.c1867383/roberta-tokenizer' --model-path='/scratch/c.c1867383/roberta-for-sequence-classification' --results-path='/results'
#finetuning+pcl_setting4.py --dataset-path='new_setting/data_exp_all_binarized.csv' --times-negs=3000 --tokenizer-path='/scratch/c.c1867383/roberta-tokenizer' --model-path='/scratch/c.c1867383/roberta-for-sequence-classification' --results-path='/results'
#nohup python3 -u cats_pcl_adapters_newsetting.py --dataset-path='new_setting/binary_cats_dataset_The_poorer_the_merrier.csv' --times-negs=5

echo 'Code running'

echo 'Finished!'

###SBATCH --mem=256



