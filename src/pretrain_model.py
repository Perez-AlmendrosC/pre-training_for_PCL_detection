import torch
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
from itertools import product
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from simpletransformers.classification import MultiLabelClassificationModel,ClassificationModel
from sklearn.metrics import accuracy_score
import random
import argparse

def load_justice_sentences(data_dir, split="train"):
	path = os.path.join(data_dir, "justice_{}.csv".format(split))
	df = pd.read_csv(path)
	labels = [df.iloc[i, 0] for i in range(df.shape[0])]
	sentences = [df.iloc[i, 1] for i in range(df.shape[0])]
	df_joint=pd.DataFrame(zip(sentences, labels))
	return df_joint

# THIS CODE FOR COMMON-SENSE MORALITY DATASET
def load_cm_sentences(data_dir, split="train"):
	path = os.path.join(data_dir, "cm_{}.csv".format(split))
	df = pd.read_csv(path)
	labels = [df.iloc[i, 0] for i in range(df.shape[0])]
	sentences = [df.iloc[i, 1] for i in range(df.shape[0])]
	return df
	#return sentences, labels
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


def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(seed)
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
	#python src/jointbert/main.py --task collocations_jb --model_type distilbert-base-multilingual-cased --num_train_epochs 1 --max_seq_len 150 --train_batch_size 32 --model_dir collocations_jb_1ep_distilbert_multi_150tok --do_train --do_eval
	parser = argparse.ArgumentParser()

	parser.add_argument("--dataset-path", required=True, help="Path to ETHICS subdataset")
	parser.add_argument("--output-path", required=True, help="Output directory for storing model")
	parser.add_argument("--model-name", required=True, help="Model name (for example, 'bert')")
	parser.add_argument("--model-id", required=True, help="Model identifier (for example, 'bert-base-uncased')")
	parser.add_argument("--epochs", required=True, type=int, help="Number of epochs")

	args = parser.parse_args()

	output_dir = 'model='+args.model_name+'_epochs='+str(args.epochs)
	output_dir = os.path.join(args.output_path, output_dir)


	ethics_path = args.dataset_path
	if 'justice' in args.dataset_path:
		# THIS CODE FOR JUSTICE DATASET
		train_df = load_justice_sentences(ethics_path, split="train")
		test_hard_df = load_justice_sentences(ethics_path, split="test_hard")
		test_df = load_justice_sentences(ethics_path, split="test")    	
	elif 'commonsense' in args.dataset_path:
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
	elif 'deontology' in args.dataset_path:
		train_df = load_deontology_sentences(ethics_path, split="train")
		test_hard_df = load_deontology_sentences(ethics_path, split="test_hard")
		test_df = load_deontology_sentences(ethics_path, split="test")

	else:
		sys.exit('You must provide a valid dataset path')
	
	output_path=args.output_path
	print('loaded datasets with sizes:')
	print(len(train_df), len(test_hard_df), len(test_df))

	seed=1
	set_seed(seed)

	model = ClassificationModel(args.model_name, args.model_id,
								#use_cuda=False, 
								num_labels=2, 
								args={'reprocess_input_data': True, 
										'overwrite_output_dir': True, 
										'num_train_epochs': args.epochs, 
										'learning_rate':1e-5,  #3e-5
										'n_gpu':1, 
										'no_cache':True,
										'output_dir':output_dir,
										'train_batch_size':4, 
										'weight_decay':0.01,
										'max_seq_length':512,
										'save_model_every_epoch':False,
										#'evaluate_during_training_verbose':True, 
										#'early_stopping_consider_epochs':True, 
										#'cache_dir':output_path,
										'silent':False,
										'logging_steps':5000 
										}) 


	test_hard_accs = []
	test_accs = []

	model.train_model(train_df, output_dir=output_dir)

	th_result, th_model_outputs, th_wrong_predictions = model.eval_model(test_hard_df, test_hard_acc=accuracy_score)
	print('Hard Test Accuracy = ', th_result['test_hard_acc'])

	t_result, t_model_outputs, t_wrong_predictions = model.eval_model(test_df, test_acc=accuracy_score)
	print('Test Accuracy = ', t_result['test_acc'])

	test_hard_accs.append(th_result)
	test_accs.append(t_result)

	model.save_model()


