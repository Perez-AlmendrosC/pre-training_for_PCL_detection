import torch
import data_utils
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import random
import pandas as pd
from transformers import RobertaTokenizer, RobertaConfig, RobertaModelWithHeads, AutoTokenizer, RobertaForSequenceClassification, AutoModel
import numpy as np
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
import datasets


def encode_batch(batch):
	"""Encodes a batch of input data using the model tokenizer."""
	return tokenizer(batch["text"], max_length=512, truncation=True, padding="max_length")

def compute_accuracy(p: EvalPrediction):
	preds = np.argmax(p.predictions, axis=1)
	return {"acc": (preds == p.label_ids).mean()}


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("--dataset-name", required=True, help="Name of the dataset")
	parser.add_argument("--dataset-path", required=True, help="Path to dataset")
	parser.add_argument("--output-path", required=True, help="Output directory for storing adapter")
	parser.add_argument("--model-name", required=True, help="Model name (for example, 'bert')")
	parser.add_argument("--model-path", required=True, help="Model path")
	parser.add_argument("--num-epochs", required=True, type=int, help="Number of epochs to train the adapter")


	args = parser.parse_args()

	output_dir = 'pt_'+args.model_name+'_'+args.dataset_name
	output_dir = os.path.join(args.output_path, output_dir)

	dataset_name=args.dataset_name
	dataset_path = args.dataset_path

	dataset=datasets.load_dataset('csv', data_files=dataset_path, sep=',') #sep= '\t' for stereoset, ',' for the rest

	tokenizer = RobertaTokenizer.from_pretrained("/scratch/c.c1867383/roberta-tokenizer", do_lower_case=True)

	# Encode the input data
	dataset = dataset.map(encode_batch, batched=True)
	# Transform to pytorch tensors and only output the required columns
	dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

	print('loaded dataset with size:')
	print(len(dataset['train']))

	train_size = int(0.8 * len(dataset['train']))
	test_size = len(dataset['train']) - train_size
	train_dataset, test_dataset = torch.utils.data.random_split(dataset['train'], [train_size, test_size])

	model = RobertaForSequenceClassification.from_pretrained(args.model_path) 

	# Add a new adapter
	model.add_adapter(dataset_name)
	
	"""
	# If desired, add a matching classification head
	model.add_classification_head(dataset_name, num_labels=2)
	"""
	# Activate the adapter
	model.train_adapter(dataset_name)

	training_args = TrainingArguments(learning_rate=1e-4,
					  num_train_epochs=args.num_epochs,
					  per_device_train_batch_size=8,
					  per_device_eval_batch_size=8,
					  logging_steps=3000,
					  output_dir=args.output_path,
					  save_total_limit = 3,
					  overwrite_output_dir=True,
					  remove_unused_columns=False
					 )

	trainer = AdapterTrainer(model=model,
				 args=training_args,
				 train_dataset=train_dataset,
				 eval_dataset=test_dataset,
				 compute_metrics=compute_accuracy
				)

	print(f'Training for {args.num_epochs} epochs...')
	trainer.train()

	model.save_adapter(args.output_path, dataset_name)
	print('Adapter saved')
