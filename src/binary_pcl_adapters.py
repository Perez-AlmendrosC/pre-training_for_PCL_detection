import sys

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import time
import datetime
import os
import numpy as np
from numpy import random

from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelWithHeads

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import logging
from collections import defaultdict


def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)

def get_pretrained_name(modelpath):
  #'/scratch/c.c1867383/ethics_finetune/cm/bert-base2'
  modelname = modelpath.split('/')[-1]
  pretrain_epochs = ''
  for i in modelname:
    if i.isdigit():
      pretrain_epochs += i
  modelname = modelname.replace(pretrain_epochs, '')
  return modelname,pretrain_epochs

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
                                  truncation=True
                                  )
    inputs_ids.append(encoded_sent.get('input_ids'))
    attention_masks.append(encoded_sent.get('attention_mask'))

  inputs_ids=torch.cat(inputs_ids, dim=0)
  attention_masks=torch.cat(attention_masks, dim=0)
  labels=torch.tensor(labels_list)

  return inputs_ids, attention_masks, labels

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))      
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

  # Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def load_pcl(corpus_path):
  # load data
  corpus=[]
  with open (corpus_path) as pcl:
    for line in pcl:
      corpus.append(line)
  corpus=corpus[4:]

  print(f'Binary dataset has {len(corpus)} lines' )

  pcl_pars=[]
  pcl_labels=[]
  for line in corpus:
    t=line.strip().split('\t')[3]
    l=line.strip()[-1]
    pcl_pars.append(t)
    if l=='0' or l== '1':
      pcl_labels.append(0)
    else:
      pcl_labels.append(1)
  return pcl_pars,pcl_labels

def subsample_train (dataset, train_subsampler, number_of_negs):
  train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=train_subsampler) 
  downsample_idx=[]
  downsample_masks=[]
  downsample_labels=[]

  for line in train_dataloader:
    tks=line[0]
    masks=line[1]
    lab=line[2].item()
    if lab==1:
      downsample_idx.append(tks)
      downsample_masks.append(masks)
      downsample_labels.append(lab)
    positives=len(downsample_labels)  

  for line in train_dataloader:
    tks=line[0]
    masks=line[1]
    lab=line[2].item()
    if lab==0:
      downsample_idx.append(tks)
      downsample_masks.append(masks)
      downsample_labels.append(lab)
    if len(downsample_idx)==positives+number_of_negs:
      print(f'the downsampled dataset has now {positives} positive examples and {number_of_negs} negative examples of PCL')
      break
  c0=0
  c1=1   
  for line in downsample_labels:
    if line==0:
      c0+=1
    if line==1:
      c1+=1
  print(f'there are {c0} 0s')
  print(f'there are {c1} 1s')

  joint_data=list(zip(downsample_idx, downsample_masks,downsample_labels))
  random.shuffle(joint_data)
  downsample_idx, downsample_masks,downsample_labels = zip(*joint_data)

  down_idx=torch.cat(downsample_idx)
  down_masks=torch.cat(downsample_masks)
  down_labels=torch.tensor(downsample_labels)
  train_downsampled_data=TensorDataset(down_idx, down_masks, down_labels)
  print(len(train_downsampled_data))

  print(down_labels)
  return train_downsampled_data

def finetune_model(model, train_dataloader):
  model.train()
  for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time() # Measure how long the training epoch takes.
    total_train_loss = 0 # Reset the total loss for this epoch.
    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        # Progress update every 100 batches.
        if step % 100 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        model.zero_grad()        
        # Perform a forward pass (evaluate the model on this training batch).
        mdl = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        loss=mdl[0]
        logits=mdl[1]
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()
        # Update the learning rate.
        scheduler.step()
    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
  
  # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
  
  return model

def validate_predictions(finetuned_model, validation_dataloader, fold_counter):
  t0 = time.time()
  finetuned_model.eval()
  # Tracking variables 
  total_eval_accuracy = 0
  total_eval_fscore = 0
  total_eval_loss = 0
  nb_eval_steps = 0
  all_preds=[]
  all_true_labels=[]
  for batch in validation_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
    all_true_labels.append(b_labels)
    # Tell pytorch not to bother with constructing the compute graph during
    # the forward pass, since this is only needed for backprop (training).
    with torch.no_grad():        
      evl_mdl = finetuned_model(b_input_ids, 
                      token_type_ids=None, 
                      attention_mask=b_input_mask,
                      labels=b_labels)
      loss=evl_mdl[0]
      logits=evl_mdl[1]
      # Accumulate the validation loss.
      total_eval_loss += loss.item()
      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
      preds= np.argmax(logits, axis=1).flatten()
      labels_flat = label_ids.flatten()
      all_preds.append(preds)
  flat_all_true_labels = [item.item() for sublist in all_true_labels for item in sublist]
  flat_all_preds = [item for sublist in all_preds for item in sublist]
  #print(flat_all_true_labels)
  #print(flat_all_preds)
  print(classification_report(flat_all_true_labels, flat_all_preds))
  fold_precision = precision_score(flat_all_true_labels, flat_all_preds)
  fold_recall = recall_score(flat_all_true_labels, flat_all_preds)
  fold_f1 = f1_score(flat_all_true_labels, flat_all_preds)
  print('Results for this fold:: Prec: ',fold_precision, 'Rec: ',fold_recall, 'F1: ',fold_f1)
  results_dict[fold_counter]['p'].append(fold_precision)
  results_dict[fold_counter]['r'].append(fold_recall)
  results_dict[fold_counter]['f1'].append(fold_f1)


if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

corpus_path='dontpatronizeme_pcl.tsv'

pcl_pars,pcl_labels = load_pcl(corpus_path)

# Load the roBERTa tokenizer.
print('Loading roBERTa tokenizer...')
tokenizer = RobertaTokenizer.from_pretrained('/scratch/c.c1867383/roberta-tokenizer', do_lower_case=True) #/scratch/c.c1867383/roberta-tokenizer

# get maxlen
max_len=0
for par in pcl_pars:
  inputs=tokenizer.encode(par, add_special_tokens=True)
  # Update the maximum sentence length.
  max_len = max(max_len, len(inputs))
print('Max sentence length: ', max_len)

# tokenize and preprocess with BERT
pcl_input_ids, pcl_att_masks, pcl_labels=preprocessing_text(pcl_pars, pcl_labels)
# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(pcl_input_ids, pcl_att_masks, pcl_labels)

ALL_RESULTS_DICTS = []
# Configuration options
k_folds = 5
epochs = 10
seed=[1, 34, 42, 78, 95] #1, 34, 42, 78, 95
for s in seed:
  print(f'For seed {s}:')

  # Set fixed random number seed
  seed_val = s

  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)

  # We'll store a number of quantities such as training and validation loss, 
  # validation accuracy, and timings.
  training_stats = []

  # Measure the total training time for the whole run.
  total_t0 = time.time()


  # Define the K-fold Cross Validator
  SKFold=StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=1)  
    
  # Start print
  print('--------------------------------')


  

  # K-fold Cross Validation model evaluation
  fold_counter = 1
  results_dict = defaultdict(lambda : defaultdict(list))
  
  for fold, (train_ids, test_ids) in enumerate(SKFold.split(pcl_pars, pcl_labels)):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    # Define data loaders for training and testing data in this fold
    train_downsampled_data=subsample_train(dataset, train_subsampler,4000)

    train_dataloader = torch.utils.data.DataLoader(
                      train_downsampled_data, #dataset
                      batch_size=8) #, sampler=train_subsampler
    
    validation_dataloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=8, sampler=test_subsampler)
    
    print(f'For this fold, training is {len(train_downsampled_data)}') #train_subsampler
    print(f'For this fold, test is {len(test_subsampler)}')


    
    model=AutoModelWithHeads.from_pretrained('/scratch/c.c1867383/roberta-automodel-with-heads', #/scratch/c.c1867383/roberta-automodel-with-heads
                                              num_labels=2, 
                                              output_attentions=False, 
                                              output_hidden_states=False
                                              )

    model.add_classification_head("cosmos", num_labels=2)
    cosmos_adapter = model.load_adapter("/scratch/c.c1867383/adapters/cosmos", config="pfeiffer") #@ukp/roberta-base-cosmosqa_pfeiffer
    #socialiqa_adapter = model.load_adapter("AdapterHub/roberta-base-pf-social_i_qa", source="hf")
    model.set_active_adapters(cosmos_adapter)
    #model.set_active_adapters(socialiqa_adapter)

    # Tell pytorch to run this model on the GPU.
    model.cuda()

    optimizer = AdamW(model.parameters(),
                      lr = 2e-5, # args.learning_rate - default is 5e-5.
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                      )
    
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_downsampled_data) * epochs #train_dataloader

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps = 0, # Default value in run_glue.py
                                              num_training_steps = total_steps)

    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.
    ft_model=finetune_model(model, train_dataloader) #train_dataloader

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    validate_predictions(ft_model, validation_dataloader, fold_counter)
    fold_counter += 1

  print('For seed: ',seed_val)
  for fold in results_dict:
    for metric in results_dict[fold]:
      for res in results_dict[fold][metric]:
        print(fold,metric,'=',res)
    print('----------')
  ALL_RESULTS_DICTS.append(results_dict)

  print(ALL_RESULTS_DICTS)