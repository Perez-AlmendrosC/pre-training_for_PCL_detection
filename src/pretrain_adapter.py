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
from transformers import RobertaTokenizer, RobertaConfig, RobertaModelWithHeads
import numpy as np
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
import datasets

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("--dataset-name", required=True, help="Name of the dataset")
	parser.add_argument("--dataset-path", required=True, help="Path to dataset")
	parser.add_argument("--output-path", required=True, help="Output directory for storing adapter")
	parser.add_argument("--model-name", required=True, help="Model name (for example, 'bert')")
	parser.add_argument("--model-path", required=True, help="Model path")
	parser.add_argument("--num-epochs", required=True, type=int, help="Number of epochs to train the adapter")


	args = parser.parse_args()



	def encode_batch(batch):
		"""Encodes a batch of input data using the model tokenizer."""
		return tokenizer(batch["text"], max_length=512, truncation=True, padding="max_length")

	def compute_accuracy(p: EvalPrediction):
		preds = np.argmax(p.predictions, axis=1)
		return {"acc": (preds == p.label_ids).mean()}



	output_dir = 'pt_'+args.model_name+'_'+args.dataset_name
	output_dir = os.path.join(args.output_path, output_dir)

	dataset_name=args.dataset_name
	dataset_path = args.dataset_path

	dataset=datasets.load_dataset('csv', data_files=dataset_path)

	tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)

	# Encode the input data
	dataset = cm.map(encode_batch, batched=True)
	# The transformers model expects the target class column to be named "labels"
	#dataset.rename_column_("label", "labels")
	# Transform to pytorch tensors and only output the required columns
	dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

	print('loaded dataset with size:')
	print(len(dataset['train']))

	train_size = int(0.8 * len(dataset['train']))
	test_size = len(dataset['train']) - train_size
	train_dataset, test_dataset = torch.utils.data.random_split(dataset['train'], [train_size, test_size])

	config = RobertaConfig.from_pretrained("roberta-base", num_labels=2)

	model = RobertaModelWithHeads.from_pretrained("roberta-base", config=config)

	# Add a new adapter
	model.add_adapter(dataset_name)
	# Add a matching classification head
	model.add_classification_head(dataset_name, num_labels=2)
	# Activate the adapter
	model.train_adapter(dataset_name)

	training_args = TrainingArguments(
									learning_rate=1e-5,
									num_train_epochs=args.num_epochs,
									per_device_train_batch_size=8,
									per_device_eval_batch_size=8,
									logging_steps=5000,
									output_dir='.',
									overwrite_output_dir=True,
									# The next line is important to ensure the dataset labels are properly passed to the model
									remove_unused_columns=False
									)

	trainer = AdapterTrainer(model=model,
							args=training_args,
							train_dataset=train_dataset,
							eval_dataset=test_dataset,
							compute_metrics=compute_accuracy
							)

	print(f'Training for {num_train_epochs} epochs...')
	trainer.train()

	print(f'Evaluating...')
	trainer.evaluate()



	model.save_adapter(args.output_path, dataset_name)
	print('Adapter saved')