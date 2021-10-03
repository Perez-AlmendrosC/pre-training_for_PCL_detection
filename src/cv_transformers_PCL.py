import sys
sys.path.append('lib/')
import torch
import torch.nn as nn
import os
import numpy as np
from numpy import random
#from transformers import AutoTokenizer
#from transformers import AutoModelForSequenceClassification, AdamW, BertConfig
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold,StratifiedKFold

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

if __name__ == '__main__':

  args = sys.argv[1:]


  if len(args) == 3:
    
    # define data paths and global vars
    #cat_path='/content/drive/MyDrive/Colab Notebooks/PRETRAINING/datasets/DPM/dontpatronizeme_categories.tsv'
    #corpus_path='/content/drive/MyDrive/Colab Notebooks/PRETRAINING/datasets/DPM/dontpatronizeme_pcl.tsv'
    #out_model='pt_models'
    #results_path='results'

    cat_path = args[0]
    corpus_path = args[1]
    # we need to specify local path
    model_path = args[2]
    

    # load data
    corpus=[]
    with open (corpus_path) as pcl:
      for line in pcl:
        corpus.append(line)
    corpus=corpus[4:]

    categories=[]

    with open (cat_path) as categs:
      for line in categs:
        categories.append(line)
    categories=categories[4:]

    """# Binary classification"""

    print(f'Binary dataset has {len(corpus)} lines' )

    ### Binary PCL dataset ###

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

    pcl_df=pd.DataFrame(list(zip(pcl_pars, pcl_labels)), columns=['paragraph', 'label'])

    pcl_df.head()

    # Load the roBERTa tokenizer.
    print('Loading roBERTa tokenizer...')
    tokenizer = RobertaTokenizer.from_pretrained(model_path, do_lower_case=True)

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

    max_len=0

    for par in pcl_pars:
      inputs=tokenizer.encode(par, add_special_tokens=True)


      # Update the maximum sentence length.
      max_len = max(max_len, len(inputs))

    print('Max sentence length: ', max_len)

    pcl_input_ids, pcl_att_masks, pcl_labels=preprocessing_text(pcl_pars, pcl_labels)

    # Combine the training inputs into a TensorDataset.

    from torch.utils.data import TensorDataset
    dataset = TensorDataset(pcl_input_ids, pcl_att_masks, pcl_labels)

    import time
    import datetime

    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    import numpy as np

    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    #train_dataloader

    # Configuration options
    k_folds = 5
    epochs = 10


    # For fold results
    results = {}

    # Set fixed random number seed
    seed_val = 42

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
    kfold = KFold(n_splits=k_folds, shuffle=True)
      
    # Start print
    print('--------------------------------')


    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
      
      # Print
      print(f'FOLD {fold}')
      print('--------------------------------')
      
      # Sample elements randomly from a given list of ids, no replacement.
      train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)

      

      validation_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
      
      # Define data loaders for training and testing data in this fold
      train_dataloader = torch.utils.data.DataLoader(dataset, sampler=train_subsampler)
      
      downsample_idx=[]
      downsample_masks=[]
      downsample_labels=[]

      for line in train_dataloader:
        tks=line[0]
        masks=line[1]
        lab=line[2]
        if lab==1:
          downsample_idx.append(tks)
          downsample_masks.append(masks)
          downsample_labels.append(lab)

        if lab==0:
          if len(downsample_idx)<4002:
            downsample_idx.append(tks)
            downsample_masks.append(masks)
            downsample_labels.append(lab)
          else:
            print('the downsampled dataset has now 1002 positive examples and 3000 negative examples of PCL')
            break

      down_idx=torch.cat(downsample_idx)
      down_masks=torch.cat(downsample_masks)
      down_labels=torch.tensor(downsample_labels)
      train_downsampled_data=TensorDataset(down_idx, down_masks, down_labels)



      downsampled_train_dataloader = torch.utils.data.DataLoader(
                        train_downsampled_data, 
                        batch_size=8)#, sampler=train_downsampled_data
      
      validation_dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=8, sampler=validation_subsampler)
      
      print(f'For this fold, training is {len(train_downsampled_data)}')
      print(f'For this fold, validation is {len(validation_subsampler)}')
      

      model=RobertaForSequenceClassification.from_pretrained(model_path, 
                                                               num_labels=2, 
                                                               output_attentions=False, 
                                                               output_hidden_states=False
                                                               )
      # Tell pytorch to run this model on the GPU.
      model.cuda()

      optimizer = AdamW(model.parameters(),
                        lr = 2e-5, # args.learning_rate - default is 5e-5.
                        eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                        )
      


      # Total number of training steps is [number of batches] x [number of epochs]. 
      # (Note that this is not the same as the number of training samples).
      total_steps = len(train_dataloader) * epochs

      # Create the learning rate scheduler.
      from transformers import get_linear_schedule_with_warmup

      scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
      model.train()

      for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        
        # For each batch of training data...
        for step, batch in enumerate(downsampled_train_dataloader):

            # Progress update every 100 batches.
            if step % 100 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(downsampled_train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

     
            model.zero_grad()        

     
            mdl = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
            loss=mdl[0]
            logits=mdl[1]

     
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
        avg_train_loss = total_train_loss / len(downsampled_train_dataloader)            
      
      # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
          
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

      print("")
      print("Running Validation...")

      t0 = time.time()

      # Put the model in evaluation mode--the dropout layers behave differently
      # during evaluation.
      model.eval()

      # Tracking variables 
      total_eval_accuracy = 0
      total_eval_fscore = 0
      total_eval_loss = 0
      nb_eval_steps = 0
      all_preds=[]
      all_true_labels=[]

      # Evaluate data for one epoch
      for batch in validation_dataloader:
          

          b_input_ids = batch[0].to(device)
          b_input_mask = batch[1].to(device)
          b_labels = batch[2].to(device)
          all_true_labels.append(b_labels)
          # Tell pytorch not to bother with constructing the compute graph during
          # the forward pass, since this is only needed for backprop (training).
          with torch.no_grad():

            evl_mdl = model(b_input_ids, 
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

          preds= np.argmax(logits, axis=1).flatten() #
          #preds=torch.cat(preds)
          labels_flat = label_ids.flatten()
          all_preds.append(preds)

          # Calculate the accuracy for this batch of test sentences, and
          # accumulate it over all batches.
          total_eval_accuracy += np.sum(preds == labels_flat) / len(labels_flat)

          #Calculate the F-score for this batch of test sentences, and 
          #accumulate it over all batches.
          total_eval_fscore += f1_score(labels_flat, preds)
          
      # Report the final accuracy for this validation run.
      avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
      print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

      # Report the final F-score for this validation run.
      avg_val_fscore = total_eval_fscore / len(validation_dataloader)
      print("  F-score: {0:.2f}".format(avg_val_fscore))

      # Calculate the average loss over all of the batches.
      avg_val_loss = total_eval_loss / len(validation_dataloader)
      
      # Measure how long the validation run took.
      validation_time = format_time(time.time() - t0)
      
      print("  Validation Loss: {0:.2f}".format(avg_val_loss))
      print("  Validation took: {:}".format(validation_time))

      # Record all statistics from this epoch.
      training_stats.append(
          {
              'epoch': epoch_i + 1,
              'Training Loss': avg_train_loss,
              'Valid. Loss': avg_val_loss,
              'Valid. Accur.': avg_val_accuracy,
              'Valid. F-score': avg_val_fscore,
              'Training Time': training_time,
              'Validation Time': validation_time
          }
      )

      flat_all_true_labels = [item.item() for sublist in all_true_labels for item in sublist]
      flat_all_preds = [item for sublist in all_preds for item in sublist]
      results=classification_report(flat_all_preds, flat_all_true_labels)
      print(results)

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

