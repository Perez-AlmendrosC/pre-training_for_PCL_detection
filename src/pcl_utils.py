import pandas as pd
import torch
import torch.nn as nn

import pytorch_metric_learning
from torch.utils.data import TensorDataset
#from pytorch_metric_learning import losses
import time
import datetime
import os
import numpy as np
from numpy import random

from transformers import RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, AdamW, AutoTokenizer, DistilBertForSequenceClassification, RobertaTokenizerFast, DebertaTokenizer, DebertaForSequenceClassification
#from transformers import Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from collections import defaultdict

# Check if GPU is available
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def unify_data(binary_df, categories_df):
    # FIRST
    #keep paragraphs and binarized categories labels from the categories_df data:
    cats=categories_df[['text', 'label']]
    # Add the binary label for the paragraph. As all the paragraphs which contain categories are positive examples of PCL
    # the binary label will be 1. 
    bin_labels_pos=[1 for i in range(len(cats))]
    # Insert a column with the binary label values and call it bin_label.
    cats.insert(2, 'bin_label', bin_labels_pos)
    pos_binary=cats.rename(columns={'label':'cats_label'})

    # SECOND
    # Take the negative paragraphs from the binary_df data:
    neg_binary=binary_df[binary_df.label==0]
    # Create a categories' label for each negative instance, which will be a numpy array of 7 zeros.
    bin_labels_neg=[]
    for i in range(len(neg_binary)):
      bin_labels_neg.append(np.zeros(7, dtype=int))
    # Include the new binarized categories label (negative cases) in the neg_binary dataframe  
    neg_binary.insert(5,'binarized_labels', bin_labels_neg)
    # Keep just the paragraph, binarized categories label and binary label from the neg_binary dataframe.
    neg_binary=neg_binary[['text', 'binarized_labels', 'label']]
    # and rename the new column to 'cats_labels':
    neg_binary = neg_binary.rename(columns={'binarized_labels':'cats_label', 'label':'bin_label'})

    # THIRD
    # Concatenate positive and negative cases' dataframes into one
    pcl_data=pd.concat([neg_binary, pos_binary])
    # Convert the categories' label into a list
    pcl_data.cats_label = pcl_data.cats_label.apply(strlabels2list)

    return pcl_data





  # Test the finetuned model on the test set: 



  # Create validation setting:

def create_validation_setting(train_dataset):
    """
    This function creates a validation setting to be used if validation while training is desired.
    It takes as input the train dataset and splits it into a 80% training (tr) and 20% validations (v) for each fold 
    """
    tr_indices, v_indices= train_test_split(range(len(train_dataset)),
                                            test_size=0.20,
                                            random_state=1
                                            )
    tr=[]
    for idx in t_indices:
      tr.append(train_dataset[idx])
    v=[]
    for idx in v_indices:
      v.append(train_dataset[idx])
    return tr, v




def preprocessing_text(paragraphs_list, labels_list, tokenizer):
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

    inputs_ids, attention_masks, labels= torch.cat(inputs_ids, dim=0), torch.cat(attention_masks, dim=0), torch.tensor(labels_list)  

    return inputs_ids, attention_masks, labels

def finetune_model(model, train_dataloader, epochs, optimizer, scheduler, device, validation_dataloader=None, setting='binary'):

    model.to(device)
    model.train()

    for epoch_i in range(epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_train_loss = 0

        with torch.set_grad_enabled(True):

            for step, batch in enumerate(train_dataloader):

                if step % 100 == 0 and not step == 0:

                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                if setting=='multilabel':
                    b_labels = batch[2].to(device, dtype = torch.float)
                else:
                    b_labels = batch[2].to(device)

                optimizer.zero_grad()
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[0]
                logits = outputs[1]
                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                logits = logits.detach().cpu().numpy()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        if validation_dataloader:
            model.eval()
            total_eval_loss = 0
            total_samples = 0
            all_preds = []
            all_true_labels = []

            with torch.no_grad():
                for batch in validation_dataloader:
                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)

                    if setting=='multilabel':
                        b_labels = batch[2].to(device, dtype = torch.float)
                    else:
                        b_labels = batch[2].to(device)
                    all_true_labels.append(b_labels)

                    evl_outputs = model(b_input_ids, attention_mask=b_input_mask)
                    logits = evl_outputs[0]
                    loss = torch.nn.functional.cross_entropy(logits, b_labels)

                    total_eval_loss += loss.item()
                    total_samples += b_labels.size(0)

                    logits = logits.detach().cpu().numpy()
                    preds = np.argmax(logits, axis=1).flatten()
                    all_preds.append(preds)

            flat_all_true_labels = [item.item() for sublist in all_true_labels for item in sublist]
            flat_all_preds = [item for sublist in all_preds for item in sublist]
            res = classification_report(flat_all_true_labels, flat_all_preds, digits=4)
            print(res)

            avg_eval_loss = total_eval_loss / total_samples
            print("  Average evaluation loss: {0:.2f}".format(avg_eval_loss))
    return model


def test_predictions(finetuned_model, test_dataloader, device, setting='binary'):

    """
    This functions takes as imput the finetuned model resulting from the finetune_model function, 
    the test_dataloder and the fold counter (based on cross-validation, for information purposes). 
    This function prints and saves into a results.txt files the results for each fold.
    """

    finetuned_model.eval()
    total_eval_loss = 0
    all_preds=[]
    all_true_labels=[]

    with torch.no_grad():  
        for batch in test_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            if setting=='multilabel':
                b_labels = batch[2].to(device, dtype = torch.float)
            else: 
                b_labels = batch[2].to(device)
                all_true_labels.append(b_labels)               
             
            evl_outputs = finetuned_model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            #loss = torch.nn.functional.cross_entropy(evl_outputs.logits, b_labels)

            loss=evl_outputs[0]
            logits=evl_outputs[1]
            total_eval_loss += loss.item()

            if setting=='binary':

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                preds= np.argmax(logits, axis=1).flatten()
                all_preds.append(preds)

            else:
                all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
                all_true_labels.extend(b_labels.to('cpu').numpy().tolist())

        if setting=='binary':
            flat_all_true_labels = [item.item() for sublist in all_true_labels for item in sublist]
            flat_all_preds = [item for sublist in all_preds for item in sublist]
            res=classification_report(flat_all_true_labels, flat_all_preds, digits=4)
        else:
            all_preds=np.array(all_preds)>=0.5
            #print('all_preds: ',len(all_preds))
            #print('all_true_labels: ',len(all_true_labels))
            res=classification_report(all_true_labels, all_preds, digits=4)
        
    return res











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
    c_pos=0
    c_negs+=1
    for line in train_dataloader:
        to
    if lab ==1:
        downsample_idx.append(tks)
        downsample_masks.append(masks)
        downsample_labels.append(lab)
        c_pos+=1
    if lab==0:
        if c_negs==n_negs:
            pass
        else:
            downsample_idx.append(tks)
            downsample_masks.append(masks)
            downsample_labels.append(lab)
            c_negs+=1
      
    #len(downsample_idx)==positives+n_negs:
    print(f'the downsampled dataset has now {c_pos} positive examples and {c_negs} negative examples of PCL')


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


def test_loss_fn(outputs, targets):
    loss_func=torch.nn.BCEWithLogitsLoss()
    return loss_func(outputs, targets)


def loss_fn(outputs, targets):
    loss_func = losses.NTXentLoss()
    return loss_func(outputs, targets)


def create_training_set_CV(pos_data, neg_data, train_ids_list, num_negs, setting='binary'):
    """
    This function creates the training set with all positive examples corresponding to the train indexes given by a CV setting and n negative examples. 
    It takes as input: a df with pos_data, a df with neg_data, the 
    train_ids_list from a train-test split setting, the desired number of negatives to include, and the setting ('binary' or 'multilabel'). 
    The setting is set to binary by default. Specify 'multilabel' in a multilabel classification problem
    """
    train=[]
    for idx in train_ids_list:
        train.append(pos_data[idx])

    
    if setting == 'binary':
        train=train+neg_data[:num_negs]
        random.seed(5)
        random.shuffle(train)
        print('Training set for this fold has ',len(train), ' samples')
        train_sentences=[i[0] for i in train]
        train_labels=[i[2] for i in train]
    if setting=='multilabel':
        train=train+neg_data[:num_negs]
        random.seed(5)
        random.shuffle(train)
        print('Training set for this fold has ',len(train), ' samples')
        train_sentences=[i[0] for i in train]
        train_labels=[i[1] for i in train]
    # Prepare the data  
    return train_sentences, train_labels

  
  

def create_training_set_train_test(pos_data, neg_data, num_negs, setting='binary'):  
    train=[]
    for line in pos_data:
        train.append(line)
    train=train+neg_data[:num_negs]
    random.seed(5)
    random.shuffle(train)
    print('Training set has ',len(train), ' samples')
    train_sentences=[i[0] for i in train]
    print(train_sentences[:5])
    if setting == 'binary':
        train_labels=[i[2] for i in train]
    if setting=='multilabel':
        train_labels=[i[1] for i in train]
    # Prepare the data  
    return train_sentences, train_labels

    """
    train_input_ids, train_att_masks, train_labels=preprocessing_text(train_sentences, train_labels)
    # Combine the training inputs into a TensorDataset.
    train_dataset = TensorDataset(train_input_ids, train_att_masks, train_labels)
    """
  

def create_raw_test_set(pos_data, test_negs, test_ids_list):
  test=[]
  for idx in test_ids_list:
    test.append(pos_data[idx])
  test=test+test_negs
  random.seed(5)
  random.shuffle(test)
  print('Test set for this fold has ',len(test), ' samples')
  return test
  
def create_test_set(pos_data, neg_data, test_ids_list, setting='binary'):
    """
    This function creates the test set with all positive and negative examples corresponding to 
    the test split for this fold. It takes as input: a df with pos_data, a df with neg_data, the 
    test_ids_list from a train-test split and the setting ('binary' or 'multilabel'). 
    The setting is set to binary by default. Specify 'multilabel' in a multilabel classification problem
    """
    test=[]
    for idx in test_ids_list:
        test.append(pos_data[idx])

    if setting == 'binary':
        test=test+neg_data
        random.seed(5)
        random.shuffle(test)
        print('Test set for this fold has ',len(test), ' samples')
        test_sentences=[i[0] for i in test] 
        test_labels=[i[2] for i in test]
        
    if setting=='multilabel':
        print('Test set for this fold has ',len(test), ' samples')
        test_sentences=[i[0] for i in test] 
        test_labels=[i[1] for i in test]

    return test_sentences, test_labels

 

def strlabels2list(strlabels):
  return [int(k) for k in strlabels] #[1:-1].split()