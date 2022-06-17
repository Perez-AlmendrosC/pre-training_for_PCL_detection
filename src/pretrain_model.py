
import simpletransformers
import torch
import os
import numpy as np
from simpletransformers.classification import MultiLabelClassificationModel,ClassificationModel
import random
import pandas as pd
import datasets

# Check if GPU is available
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)

def load_dataset(dataset_path):
	dataset=datasets.load_dataset('csv', data_files=dataset_path, sep=',')
	return dataset


# Load dataset

train_df=pd.read_csv(dataset_path) #, change to sep='\t' for stereoset dataset
print('loaded dataset with size:')
print(len(train_df))

# Prepare model and train

MODEL_NAME='roberta'
MODEL_ID='roberta-base'
EPOCHS=5
LR=1e-5
dataset_path='/content/cm_clean.csv'
output_dir = '.'


seed=1
set_seed(seed)

model = ClassificationModel(MODEL_NAME, MODEL_ID,
                            num_labels=2, 
                            args={'reprocess_input_data': True, 
                                  'overwrite_output_dir': True, 
                                  'num_train_epochs': EPOCHS, 
                                  'learning_rate':LR,
                                  'n_gpu':1, 
                                  'output_dir':output_dir,
                                  'train_batch_size':8, #changed to 8 as adapters training. 
                                  'weight_decay':0.01,
                                  'max_seq_length':512,
                                  'dataloader_num_workers': 2,
                                  'save_model_every_epoch':False,
                                  'no_cache':True,
                                  'silent':False,
                                  'logging_steps':3000
                                  }) 


model.train_model(train_df, output_dir=output_dir)

model.save_model()

