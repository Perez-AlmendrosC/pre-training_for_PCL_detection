

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
import time
import datetime
import numpy as np
from numpy import random


from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, AutoTokenizer, AutoModelWithHeads, RobertaTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from collections import defaultdict
from argparse import ArgumentParser


import sys
import os
from pcl_utils import *

# Check if GPU is available
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Define functions for finetuning and evaluating

def finetune_model(model, train_dataloader, epochs, validation_dataloader=None):
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
        b_labels = batch[2].to(device) # for multilabel --> , dtype = torch.float
        model.zero_grad()        
        # Perform a forward pass (evaluate the model on this training batch).
        outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask,
                    labels=b_labels
                    )
              
        loss=outputs[0]
        logits=outputs[1]
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        
        #loss = loss_fn(logits.view(-1,num_labels),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
        total_train_loss += loss.item()
        #total_train_loss += loss_fn(outputs, b_labels)
        
        optimizer.zero_grad()
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
    print("  Training epoch took: {:}".format(training_time))
    
    # Validate while training
    if validation_dataloader:
      total_eval_accuracy = 0
      total_eval_fscore = 0
      total_eval_loss = 0
      nb_eval_steps = 0
      all_preds=[]
      all_true_labels=[]
      model.eval()
      with torch.no_grad():  
        for batch in validation_dataloader:
          b_input_ids = batch[0].to(device)
          b_input_mask = batch[1].to(device)
          b_labels = batch[2].to(device)# for multilabel --> , dtype = torch.float
          all_true_labels.append(b_labels)
          # Tell pytorch not to bother with constructing the compute graph during
          # the forward pass, since this is only needed for backprop (training).
                
          evl_outputs = model(b_input_ids, 
                              token_type_ids=None, 
                              attention_mask=b_input_mask,
                              labels=b_labels
                              )

          loss=evl_outputs[0]
          logits=evl_outputs[1]
        
          # Accumulate the validation loss.       
          #loss = loss_fn(logits.view(-1,num_labels),b_labels.type_as(logits).view(-1,num_labels))
          total_eval_loss += loss.item()

          # for binary classification:  
          # Move logits and labels to CPU
          logits = logits.detach().cpu().numpy()
          #label_ids = b_labels.to('cpu').numpy()
          preds= np.argmax(logits, axis=1).flatten()
          #labels_flat = label_ids.flatten()
          all_preds.append(preds)
        flat_all_true_labels = [item.item() for sublist in all_true_labels for item in sublist]
        flat_all_preds = [item for sublist in all_preds for item in sublist]
        res=classification_report(flat_all_true_labels, flat_all_preds, digits=4)
        print(res)

  return model


def test_predictions(finetuned_model, test_dataloader, fold_counter):

  #with open(os.path.join('results',f'results_full_finetuning_CM2_binary_2knegs.txt'), 'a')as outf:
  with open(os.path.join('results',f'results_adapter={args.adapter_name}_rec_binary.txt'), 'a')as outf:
  #with open(os.path.join('results',f'results_baseline_rec_binary.txt'), 'a')as outf:

    outf.write(f'For run {s}: \n')
    outf.write(f'For fold {fold}: \n')
    t0 = time.time()
    finetuned_model.eval()
    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_fscore = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    all_preds=[]
    all_true_labels=[]
    with torch.no_grad():  
      for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)# for multilabel --> , dtype = torch.float
        all_true_labels.append(b_labels)
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
              
        evl_outputs = finetuned_model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask,
                        labels=b_labels
                        )

        loss=evl_outputs[0]
        logits=evl_outputs[1]
      
        # Accumulate the validation loss.       
        #loss = loss_fn(logits.view(-1,num_labels),b_labels.type_as(logits).view(-1,num_labels))
        total_eval_loss += loss.item()

      # for binary classification:  
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        #label_ids = b_labels.to('cpu').numpy()
        preds= np.argmax(logits, axis=1).flatten()
        #labels_flat = label_ids.flatten()
        all_preds.append(preds)
      flat_all_true_labels = [item.item() for sublist in all_true_labels for item in sublist]
      flat_all_preds = [item for sublist in all_preds for item in sublist]
      res=classification_report(flat_all_true_labels, flat_all_preds, digits=4)
      print(res)
      outf.write(res)
      print(res)

      raw_gold=[]
      raw_preds=[]
      
      for index, (i_t, i_p) in enumerate(zip(flat_all_true_labels, flat_all_preds)):
        raw_gold.append(raw_test_dataset[index])
        raw_preds.append(flat_all_preds[index])

     
      fold_precision = precision_score(flat_all_true_labels, flat_all_preds)
      fold_recall = recall_score(flat_all_true_labels, flat_all_preds)
      fold_f1 = f1_score(flat_all_true_labels, flat_all_preds)
      print('Results for this fold:: Prec: ',fold_precision, 'Rec: ',fold_recall, 'F1: ',fold_f1)
      results_dict[fold_counter]['p'].append(fold_precision)
      results_dict[fold_counter]['r'].append(fold_recall)
      results_dict[fold_counter]['f1'].append(fold_f1)


      return raw_gold, raw_preds

def populate_recall_dicts(cats_df):
  # cats_df - Dataframe with gold labels in test set and 
  # predictions for each fold
  for cat_name, numb_pos in dict(cats_df.sum()).items():
    if cat_name != 'bin_preds':
      all_gold_cats[cat_name].append(numb_pos)
  for idx in range(0,7):
    tps = len(cats_df[(cats_df[idx] == 1) & (cats_df.bin_preds == 1)])
    results[idx].append(tps)





if __name__ == '__main__':

  parser = ArgumentParser()

  parser.add_argument("--dataset-path", required=True, help="Path to dataset")
  parser.add_argument("--times-negs", required=True, type=int, help="Number negative_samples")
  parser.add_argument("--tokenizer-path", required=True, help="Tokenizer path")
  parser.add_argument("--model-path", required=True, help="Model path")
  parser.add_argument("--adapter-name", required=True, help="Name of the adapter")
  parser.add_argument("--adapter-path", required=True, help="Adapter path")
  parser.add_argument("--results-path", required=False, help="Directory where to save results")

  args = parser.parse_args()

  from dont_patronize_me import DontPatronizeMe
  # Initialize a dpm (Don't Patronize Me) object.
  # It takes two areguments as input: 
  # (1) Path to the directory containing the training set files, which is the root directory of this notebook.
  # (2) Path to the test set, which will be released when the evaluation phase begins. In this example, 
  # we use the dataset for Subtask 1, which the code will load without labels.
  dpm = DontPatronizeMe(args.dataset_path, args.dataset_path)

  # This method loads the subtask 1 and 2 data
  dpm.load_task1()
  dpm.load_task2()
  # which we can then access as a dataframe
  bin_pcl=dpm.train_task1_df
  cats_pcl=dpm.train_task2_df

  #keep just text and binarized labels from the catgories dataframe:

  cats_exp=cats_pcl[['text', 'label']]
  bin_labels=[1 for i in range(len(cats_exp))]
  cats_exp.insert(2, 'bin_label', bin_labels)
  
  # Get just the negative examples from the binary dataset 
  bin_pcl_negs=bin_pcl[bin_pcl.label==0]

  # Convert the label to a numpy array of 7 zeros
  bin_labels=[]
  for i in range(len(bin_pcl_negs)):
      bin_labels.append(np.zeros(7, dtype=int))

  # Include the new 'binarized' label in the negatives' dataframe
  bin_pcl_negs.insert(5,'binarized_labels', bin_labels)

# Keep just text and binarized labels from the negatives dataframe and rename the new column to 'label':

  negs_exp=bin_pcl_negs[['text', 'binarized_labels', 'label']]
  negs_exp = negs_exp.rename(columns={'binarized_labels':'label', 'label':'bin_label'})

# Concatenate categories and negative dataframes into one

  data_exp=pd.concat([negs_exp, cats_exp])
  #data_exp=data_exp.sample(frac=1)
  data_exp.label = data_exp.label.apply(strlabels2list)

  # Get positive and negative examples into two different dataframes and make lists with paragraphs and labels
  data_pos=data_exp[data_exp.bin_label==1]
  data_neg=data_exp[data_exp.bin_label==0]

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





  # Load the roBERTa tokenizer.
  print('Loading roBERTa tokenizer...')
  tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_path, do_lower_case=True)



  # Fine-tune - Train, validate per epoch and test by fold


  NUM_LABELS=2
  TIMES_NEGS=args.times_negs
  K_FOLDS = 5
  EPOCHS = 5
  seed=[1,34, 49,78, 95]    #, 34, 49,78, 95
  adapter_name=args.adapter_name

  for s in seed:
    seed_val = s
    np.random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    print(f'for seed {s}:')
    training_stats = []
    total_t0 = time.time()

    kf=KFold(n_splits=K_FOLDS, shuffle=True, random_state=1)  
      
    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    fold_counter = 1
    evaluation_dict = defaultdict(lambda : defaultdict(list))
    results_dict = defaultdict(lambda : defaultdict(list))


    results = defaultdict(list)
    all_gold_cats = defaultdict(list)

    for fold, (train_ids, test_ids) in enumerate(kf.split(positive_samples)):
      
      # Print
      print(f'FOLD {fold}')
      print('--------------------------------')
      
      # Create train data (train positives + downsampled train negatives) 
      train_dataset=create_training_set(positive_samples, train_negs, train_ids, TIMES_NEGS)

      """
      t_indices, v_indices= train_test_split(range(len(train_dataset)),
                                                            test_size=0.20,
                                                            random_state=1
                                                            )
      t=[]
      for idx in t_indices:
        t.append(train_dataset[idx])
      v=[]
      for idx in v_indices:
        v.append(train_dataset[idx])
      """

      # Create test data (test positives + test negatives) 
      raw_test_dataset=create_raw_test_set(positive_samples, test_negs, test_ids)
      test_dataset=create_test_set(positive_samples, test_negs, test_ids)


      # Define data loaders for training and testing data in this fold

      #train_downsampled_data=subsample_train(dataset, train_subsampler,times_negs)
      
      train_dataloader = torch.utils.data.DataLoader(
                        train_dataset, #t, train_dataset, #train_downsampled_data, #train_downsampled_data #dataset
                        batch_size=8  #,sampler=train_subsampler
                        )
      """
      validation_dataloader = torch.utils.data.DataLoader(
                        v, #test_dataset, #dataset,
                        batch_size=8 #, sampler=validation_subsampler
                        )
      """
      test_dataloader= torch.utils.data.DataLoader(
                        test_dataset, #dataset,
                        batch_size=8 #, sampler=validation_subsampler
                        )
      
      
      # Define the model
      model=RobertaForSequenceClassification.from_pretrained(args.model_path,
                                              num_labels=NUM_LABELS, 
                                              output_attentions=False, 
                                              output_hidden_states=False
                                              )
      
      
      #Load adapter
      adapter_name=model.load_adapter(args.adapter_path)
      model.active_adapters = adapter_name
      

      # Tell pytorch to run this model on the GPU.
      model.cuda()

      optimizer = AdamW(model.parameters(),
                        lr = 1e-5, # args.learning_rate - default is 5e-5.
                        eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                              )
            
      # Total number of training steps is [number of batches] x [number of epochs]. 
      # (Note that this is not the same as the number of training samples).
      total_steps = len(train_dataloader) * EPOCHS

      scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                  num_warmup_steps = 0, # Default value in run_glue.py
                                                  num_training_steps = total_steps)
      

      # ========================================
      #               Training
      # ========================================
      # Perform one full pass over the training set.
      #ft_model=finetune_model(model, train_dataloader, validation_dataloader, epochs=EPOCHS)
      ft_model=finetune_model(model, train_dataloader, epochs=EPOCHS)



      # ========================================
      #               Evaluate on test set
      # ========================================
      # After the completion of training, measure our performance on
      # the test set.

      print('EVALUATING ON TEST SET')

      #test_predictions(ft_model, test_dataloader, fold_counter)
      raw_gold, raw_preds=test_predictions(ft_model, test_dataloader, fold_counter)

      fold_counter += 1
      
      print('TEST: For seed: ',seed_val)
      for fold in results_dict:
        for metric in results_dict[fold]:
          for res in results_dict[fold][metric]:
            print(fold,metric,'=',res)
        print('----------')
      
        
      # ========================================
      #   Recall by category on binary results
      # ========================================
      # Extract the True Positives and
      # recall by category

      # DataFrame with the categories' labels of each one of the parag. on the test set
      cats=[]
      for line in raw_gold:
        cats.append(line[1])
      cats_df=pd.DataFrame(cats)
      # Get the total number of instances by category
      dict(cats_df.sum())

      # Add the binary label prediction for each paragraph
      bin_preds=[i for i in raw_preds]
      cats_df.insert(7, 'bin_preds', bin_preds)

      # Add the total number of instances and the TP by category for each fold
      populate_recall_dicts(cats_df)
      all_gold_cats

    # Get recall results by category for this run
    print(f'for run {s}, the recall by category is:' )
    for idx in range(0,7):
      sum_gold = sum(all_gold_cats[idx])
      sum_preds = sum(results[idx])
      print(idx)
      print(sum_preds/sum_gold)
      print('---')
