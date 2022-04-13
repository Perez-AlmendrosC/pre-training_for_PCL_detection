import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
import time
import datetime
import os
import numpy as np
from numpy import random

from transformers import BertTokenizer, RobertaTokenizer, RobertaForSequenceClassification, AdamW, AutoTokenizer, AutoModelWithHeads, RobertaTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from collections import defaultdict
from argparse import ArgumentParser

# Check if GPU is available
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



tokenizer = RobertaTokenizer.from_pretrained('/scratch/c.c1867383/roberta-tokenizer') # 'roberta-base', do_lower_case=True


def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)

def subsample_train (dataset, train_subsampler, n_negs):
  train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=train_subsampler) 
  downsample_idx=[]
  downsample_masks=[]
  downsample_labels=[]
  for line in train_dataloader:
    tks=line[0]
    masks=line[1]
    lab=line[2]#.item()
    if lab ==1:
      downsample_idx.append(tks)
      downsample_masks.append(masks)
      downsample_labels.append(lab)
    positives=len(downsample_labels)
    print(positives)
  for line in train_dataloader:
    tks=line[0]
    masks=line[1]
    lab=line[2]#.item()
    if lab==0:
      downsample_idx.append(tks)
      downsample_masks.append(masks)
      downsample_labels.append(lab)
    if len(downsample_idx)==positives+n_negs:
      print(f'the downsampled dataset has now {positives} positive examples and {n_negs} negative examples of PCL')
      break
  #count positives and negatives    
  c0=0
  c1=1   
  for line in downsample_labels:
    if line==0:
      c0+=1
    if line==1:
      c1+=1
  print(f'there are {c0} negatives')
  print(f'there are {c1} positives')
  print(len(downsample_labels))
  joint_data=list(zip(downsample_idx, downsample_masks,downsample_labels))
  random.shuffle(joint_data)
  downsample_idx, downsample_masks,downsample_labels = zip(*joint_data)
  down_idx=torch.cat(downsample_idx)
  down_masks=torch.cat(downsample_masks)
  down_labels=torch.cat(downsample_labels)
  train_downsampled_data=TensorDataset(down_idx, down_masks, down_labels)
  print('Downsample data has ', len(train_downsampled_data), ' lines')
  print(down_labels)
  return train_downsampled_data


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))      
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):
    '''
    Function to calculate the accuracy of our predictions vs labels
    '''
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def create_training_set(pos_data, neg_data, train_ids_list, num_negs):
  train=[]
  for idx in train_ids_list:
    train.append(pos_data[idx])
  train=train+neg_data[:num_negs]
  random.seed(5)
  random.shuffle(train)
  print(train[:5])
  print(train[-5:])
  print('Training set for this fold has ',len(train), ' samples')
  train_sentences=[i[0] for i in train] 
  train_labels=[i[2] for i in train] #i[1] for multilabel classification #i[2] for binary classification

  # Prepare the data  
  train_input_ids, train_att_masks, train_labels=preprocessing_text(train_sentences, train_labels)
  # Combine the training inputs into a TensorDataset.
  train_dataset = TensorDataset(train_input_ids, train_att_masks, train_labels)
  return train_dataset

def create_raw_test_set(pos_data, neg_data, test_ids_list):
  test=[]
  for idx in test_ids_list:
    test.append(pos_data[idx])
  test=test+neg_data
  random.seed(5)
  random.shuffle(test)
  print('Test set for this fold has ',len(test), ' samples')
  return test


def create_test_set(pos_data, neg_data, test_ids_list):
  test=[]
  for idx in test_ids_list:
    test.append(pos_data[idx])
  test=test+neg_data
  random.seed(5)
  random.shuffle(test)
  print(test[:5])
  print(test[-5:])
  print('Test set for this fold has ',len(test), ' samples')
  test_sentences=[i[0] for i in test] 
  test_labels=[i[2] for i in test] #i[1] for multilabel classification #i[2] for binary classification
  # Prepare the data  
  test_input_ids, test_att_masks, test_labels=preprocessing_text(test_sentences, test_labels)
  # Combine the test inputs into a TensorDataset.
  test_dataset = TensorDataset(test_input_ids, test_att_masks, test_labels)
  return test_dataset

def preprocessing_text(paragraphs_list, labels_list):
  inputs_ids=[]
  attention_masks=[]
  for par in paragraphs_list:
    encoded_sent=tokenizer.encode_plus(par, 
                                  add_special_tokens=True, 
                                  max_length=512, 
                                  pad_to_max_length = True,
                                  return_attention_mask=True,
                                  return_tensors='pt',
                                  truncation=True, 
                                  )
    inputs_ids.append(encoded_sent.get('input_ids'))
    attention_masks.append(encoded_sent.get('attention_mask'))
  inputs_ids=torch.cat(inputs_ids, dim=0)
  attention_masks=torch.cat(attention_masks, dim=0)
  labels=torch.tensor(labels_list)
  return inputs_ids, attention_masks, labels



def strlabels2list(strlabels):
  return [int(k) for k in strlabels] #[1:-1].split()