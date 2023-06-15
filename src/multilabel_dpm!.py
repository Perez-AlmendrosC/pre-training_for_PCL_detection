
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
import time
import datetime
import numpy as np
from numpy import random


from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, AdamW

from transformers import Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from collections import defaultdict
from urllib import request
import pandas as pd

from argparse import ArgumentParser

import sys
import os

"""# Check if GPU is available"""

# Check if GPU is available
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#Fetch the data


if __name__ == '__main__':

  parser = ArgumentParser()

  parser.add_argument("--dataset-path", required=True, help="Path to dataset")
  parser.add_argument("--times-negs", required=True, type=int, help="Number negative_samples")
  parser.add_argument("--tokenizer-path", required=True, help="Tokenizer path")
  parser.add_argument("--model-path", required=True, help="Model path")
  parser.add_argument("--adapter-name", required=False, help="Name of the adapter")
  parser.add_argument("--adapter-path", required=False, help="Adapter path")
  parser.add_argument("--results-path", required=False, help="Directory where to save results")

  args = parser.parse_args()

  # Load the tokenizer.
  print('Loading tokenizer...')
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

  #Fetch the data

  from pcl_utils import *
  from dont_patronize_me import DontPatronizeMe

  #Prepare data

  from dont_patronize_me import DontPatronizeMe
  # Initialize a dpm (Don't Patronize Me) object.
  # It takes two areguments as input: 

  dpm = DontPatronizeMe(args.dataset_path, args.dataset_path)

  # This method loads the subtask 1 and 2 data
  dpm.load_task1()
  dpm.load_task2()
  # which we can then access as a dataframe
  binary_df=dpm.train_task1_df
  categories_df=dpm.train_task2_df

  # Unify all data in a single dataframe
  data=unify_data(binary_df, categories_df)


  # Get positive and negative examples into two different dataframes and make lists with paragraphs and labels
  data_pos=data[data.bin_label==1]
  data_neg=data[data.bin_label==0]

  # Get positive examples in sentences and labels' list
  positive_samples=data_pos.values.tolist()
  pos_sentences=[i[0] for i in positive_samples]
  pos_multi_labels=[i[1] for i in positive_samples]
  pos_bin_labels=[i[2] for i in positive_samples]


  
  # Get negative examples in train and test split (80-20)
  negative_samples=data_neg.values.tolist()
  neg_train_indices, neg_test_indices= train_test_split(range(len(negative_samples)),
                                                      test_size=0.20,
                                                      random_state=1
                                                      )
  train_negs=[]
  for idx in neg_train_indices:
    train_negs.append(negative_samples[idx])
  test_negs=[]
  for idx in neg_test_indices:
    test_negs.append(negative_samples[idx])
  

  # Fine-tune

  NUM_LABELS=7
  TIMES_NEGS=args.times_negs
  K_FOLDS = 5
  EPOCHS = 2 #return to 10
  seed=[1,34,49,78,95]
  if args.adapter_name:
    adapter_name=args.adapter_name



  for s in seed:
  
    seed_val = s
    np.random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    print(f'for seed {s}:')
    total_t0 = time.time()

    kf=KFold(n_splits=K_FOLDS, shuffle=True, random_state=1)  
      
    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation

    #results_dict = defaultdict(lambda : defaultdict(list))

    for fold, (train_ids, test_ids) in enumerate(kf.split(positive_samples)):
      
      # Print
      print(f'FOLD {fold}')
      print('--------------------------------')
      
      # Create train data (train positives + downsampled train negatives)

      #Create train data (train positives + train negatives)
      train_sentences, train_labels=create_training_set_CV(positive_samples, train_negs, train_ids, TIMES_NEGS, setting='multilabel')

      cats_input_ids, cats_att_masks, cats_labels=preprocessing_text(train_sentences, train_labels, tokenizer=tokenizer)
      # Combine the training inputs into a TensorDataset.
      train_dataset = TensorDataset(cats_input_ids, cats_att_masks, cats_labels)


      # Create test data (test positives + test negatives)     
      test_sentences, test_labels = create_test_set(positive_samples, test_negs, test_ids, setting='multilabel')
      # Prepare the data  
      test_input_ids, test_att_masks, test_labels=preprocessing_text(test_sentences, test_labels, tokenizer=tokenizer)
      # Combine the test inputs into a TensorDataset.
      test_dataset = TensorDataset(test_input_ids, test_att_masks, test_labels)


      #train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
      #test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

      #Create raw test data:
      #raw_test=create_raw_test_set(positive_samples, test_negs, test_ids)

      # If validation while training desired, uncomment the following line.
      # tr, v= create_validation_setting(train_dataset)


      # Define data loaders for training and testing data in this fold    
      train_dataloader = torch.utils.data.DataLoader(
                        train_dataset, #change to tr if validation per fold desired
                        batch_size=4 
                        )    
      test_dataloader= torch.utils.data.DataLoader(
                        test_dataset, 
                        batch_size=4 
                        )
      # Uncomment the following lines if validation per fold is desired
      """
      validation_dataloader = torch.utils.data.DataLoader(
                        v,
                        batch_size=4
                        )
      """
      
      # Define the model
      model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=NUM_LABELS)
      
      #Load adapter
      if args.adapter_path:
        adapter=model.load_adapter(args.adapter_path)
        model.active_adapters = adapter


      optimizer = AdamW(model.parameters(),
                        lr = 1e-5, # args.learning_rate - default is 5e-5.
                        eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                              )
            
      # Total number of training steps is [number of batches] x [number of epochs]. 
      # (Note that this is not the same as the number of training samples).
      total_steps = len(train_dataloader) * EPOCHS

      scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                  num_warmup_steps = 0, 
                                                  num_training_steps = total_steps)

      # ========================================
      #               Training
      # ========================================
      # Perform one full pass over the training set.

      model=finetune_model(model, train_dataloader, epochs=EPOCHS, optimizer=optimizer, scheduler=scheduler, device=device, setting='multilabel')


      # ========================================
      #               Evaluate on test set
      # ========================================
      # After the completion of training, measure our performance on
      # the test set.

      if args.adapter_name:
        print(f'EVALUATING ON TEST SET with adapter {adapter_name}')

      res=test_predictions(model, test_dataloader, device=device, setting='multilabel')
      fold += 1

      print(res)

      del model
      torch.cuda.empty_cache()


