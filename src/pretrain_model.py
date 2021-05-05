import torch

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import numpy as np
import argparse
from itertools import product
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from simpletransformers.classification import MultiLabelClassificationModel,ClassificationModel
from sklearn.metrics import accuracy_score
import random


ethics_path=os.path.abspath("data/justice_data")
output_path='/models/justice_model/bert/5epochs'

"""
#THIS CODE FOR DEONTOLOGY DATASET

def load_deontology_sentences(data_dir, split="train"):
	path = os.path.join(data_dir, "deontology_{}.csv".format(split))
	df = pd.read_csv(path)
	labels = [df.iloc[i, 0] for i in range(df.shape[0])]
	scenarios = [df.iloc[i, 1] for i in range(df.shape[0])]
	excuses = [df.iloc[i, 2] for i in range(df.shape[0])]
	sentences = [sc + " [SEP] " + exc for (sc, exc) in zip(scenarios, excuses)]
	df_joint=pd.DataFrame(zip(sentences, labels))
	return df_joint
	#return sentences, labels

train_df = load_deontology_sentences(ethics_path, split="train")
test_hard_df = load_deontology_sentences(ethics_path, split="test_hard")
test_df = load_deontology_sentences(ethics_path, split="test")


"""
# THIS CODE FOR JUSTICE DATASET
def load_justice_sentences(data_dir, split="train"):
	path = os.path.join(data_dir, "justice_{}.csv".format(split))
	df = pd.read_csv(path)
	labels = [df.iloc[i, 0] for i in range(df.shape[0])]
	sentences = [df.iloc[i, 1] for i in range(df.shape[0])]
	df_joint=pd.DataFrame(zip(sentences, labels))
	return df_joint


train_df = load_justice_sentences(ethics_path, split="train")
test_hard_df = load_justice_sentences(ethics_path, split="test_hard")
test_df = load_justice_sentences(ethics_path, split="test")


"""
# THIS CODE FOR COMMON-SENSE MORALITY DATASET

def load_cm_sentences(data_dir, split="train"):

	path = os.path.join(data_dir, "cm_{}.csv".format(split))
	df = pd.read_csv(path)

	labels = [df.iloc[i, 0] for i in range(df.shape[0])]
	sentences = [df.iloc[i, 1] for i in range(df.shape[0])]

	return df
	#return sentences, labels


train_data = load_cm_sentences(ethics_path, split="train")
test_hard_data = load_cm_sentences(ethics_path, split="test_hard")
test_data = load_cm_sentences(ethics_path, split="test")

train_df=pd.DataFrame(train_data)
train_df=train_df[['label','input']]
train_cols=train_df.columns.tolist()
train_cols=['input', 'label']
train_df=train_df[train_cols]

test_hard_df=pd.DataFrame(test_hard_data)
test_hard_df=test_hard_df[['label','input']]
test_hard_cols=test_hard_df.columns.tolist()
test_hard_cols=['input', 'label']
test_hard_df=test_hard_df[test_hard_cols]

test_df=pd.DataFrame(test_data)
test_df=test_df[['label','input']]
test_cols=test_df.columns.tolist()
test_cols=['input', 'label']
test_df=test_df[test_cols]

"""



print('loaded datasets with sizes:')
print(len(train_df), len(test_hard_df), len(test_df))


def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(seed)
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)

seed=1

set_seed(seed)

model = ClassificationModel('bert', 'bert-base-uncased',
							#use_cuda=False, 
							num_labels=2, 
							args={'reprocess_input_data': True, 
									'overwrite_output_dir': True, 
									'num_train_epochs': 5, 
									'learning_rate':1e-5,  #3e-5
									'n_gpu':1, 
									'output_dir':output_path,
									'train_batch_size':4, 
									'weight_decay':0.01,
									'max_seq_length':512,
									'save_model_every_epoch':False,
									#'evaluate_during_training_verbose':True, 
									#'early_stopping_consider_epochs':True, 
									'cache_dir':output_path,
									'silent':False,
									'logging_steps':5000 
									}) 



test_hard_accs = []
test_accs = []



model.train_model(train_df, output_dir=output_path)

th_result, th_model_outputs, th_wrong_predictions = model.eval_model(test_hard_df, test_hard_acc=accuracy_score)
print('Hard Test Accuracy = ', th_result['test_hard_acc'])

t_result, t_model_outputs, t_wrong_predictions = model.eval_model(test_df, test_acc=accuracy_score)
print('Test Accuracy = ', t_result['test_acc'])


test_hard_accs.append(th_result)
test_accs.append(t_result)

model.save_model()


