
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import time
import datetime
import os
import numpy as np
from numpy import random

from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, AutoTokenizer, AutoModelWithHeads, RobertaTokenizerFast
from transformers import get_linear_schedule_with_warmup

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


def load_data(corpus_path):
  # load data
  corpus=[]
  with open (corpus_path) as pcl:
    for line in pcl:
      corpus.append(line)
  corpus=corpus[4:]
  return corpus

def strlabels2list(strlabels):
  return [int(k) for k in strlabels[1:-1].split()]

def subsample_train (dataset, train_subsampler, n_negs):
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
    if len(downsample_idx)==positives+(positives*n_negs):
      print(f'the downsampled dataset has now {positives} positive examples and {positives*n_negs} negative examples of PCL')
      break
  #count positives and negatives    
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
  print('Downsample data has ', len(train_downsampled_data), ' lines')

  print(down_labels)
  return train_downsampled_data

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


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

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
        b_labels = batch[2].to(device, dtype = torch.float) 
        model.zero_grad()        
        # Perform a forward pass (evaluate the model on this training batch).
        outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask)
                    #labels=b_labels)
              
        #loss=outputs[0]
        logits=outputs[0]
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        loss = loss_fn(logits.view(-1,num_labels),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
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
  with torch.no_grad():  
    for batch in validation_dataloader:
      b_input_ids = batch[0].to(device)
      b_input_mask = batch[1].to(device)
      b_labels = batch[2].to(device, dtype = torch.float)#, dtype = torch.float
      # Tell pytorch not to bother with constructing the compute graph during
      # the forward pass, since this is only needed for backprop (training).
            
      evl_outputs = finetuned_model(b_input_ids, 
                      token_type_ids=None, 
                      attention_mask=b_input_mask)
                      #labels=b_labels)
      logits=evl_outputs[0]

      all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
      all_true_labels.extend(b_labels.to('cpu').numpy().tolist())

  all_preds=np.array(all_preds)>=0.5
  print('all_preds: ',len(all_preds))
  print('all_true_labels: ',len(all_true_labels))
  print(classification_report(all_true_labels, all_preds, digits=4))
     


if __name__ == '__main__':

  parser = ArgumentParser()

  parser.add_argument("--dataset-path", required=True, help="Path to dataset")
 # parser.add_argument("--times-negs", required=True, type=int, help="Number of negatives")
  parser.add_argument("--tokenizer-path", required=True, help="Tokenizer path")
  parser.add_argument("--model-path", required=True, help="Model path")
  parser.add_argument("--results-path", required=False, help="Directory where to save results")


  args = parser.parse_args()


  if torch.cuda.is_available():       
      device = torch.device("cuda")
      print(f'There are {torch.cuda.device_count()} GPU(s) available.')
      print('Device name:', torch.cuda.get_device_name(0))

  else:
      print('No GPU available, using the CPU instead.')
      device = torch.device("cpu")


  dataset_path= args.dataset_path #'/content/drive/MyDrive/Colab_Notebooks/PRETRAINING/datasets/all_binarized.csv'
  data=pd.read_csv(dataset_path, sep='\t')



  strlabels2list(data.labels.iloc[0])

  data.labels = data.labels.apply(strlabels2list)

  pars=data['text'].values.tolist()
  labels=data['labels'].values.tolist()

  # Load the roBERTa tokenizer.
  print('Loading roBERTa tokenizer...')
  tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

  # tokenize and preprocess with roBERTa
  cats_input_ids, cats_att_masks, cats_labels=preprocessing_text(pars, labels)

  # Combine the training inputs into a TensorDataset.
  dataset = TensorDataset(cats_input_ids, cats_att_masks, cats_labels)


  num_labels=7
  #times_negs=args.times_negs
  k_folds = 5
  epochs = 10

  results_per_run = []

  seed=[1, 34, 49, 78, 95] #  
  for s in seed:
    seed_val = s
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    print(f'for seed {s}:')
    training_stats = []
    total_t0 = time.time()

    kf=KFold(n_splits=k_folds, shuffle=True, random_state=1)  
      
    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    fold_counter = 1
    results_dict = defaultdict(lambda : defaultdict(list))

    for fold, (train_ids, test_ids) in enumerate(kf.split(dataset)):
      
      # Print
      print(f'FOLD {fold}')
      print('--------------------------------')
      
      # Sample elements randomly from a given list of ids, no replacement.
      
      train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
      validation_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

      # Define data loaders for training and testing data in this fold

      if args.times_negs:

        train_downsampled_data=subsample_train(dataset, train_subsampler,times_negs)

        train_dataloader = torch.utils.data.DataLoader(
                        train_downsampled_data, #dataset
                        batch_size=8) #, sampler=train_subsampler

        validation_dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=8, sampler=validation_subsampler)
      
      else: 

        train_dataloader = torch.utils.data.DataLoader(
                          dataset, #train_downsampled_data
                          batch_size=8, sampler=train_subsampler)
      
        validation_dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=8, sampler=validation_subsampler)
      
      print(f'For this fold, training is {len(train_dataloader)}')
      print(f'For this fold, test is {len(validation_dataloader)}')

      model=RobertaForSequenceClassification.from_pretrained('roberta-base', 
                                               num_labels=num_labels, 
                                               output_attentions=False, 
                                               output_hidden_states=False
                                               )

      """
      #commonsense_morality

      commonsense10=model.load_adapter("/content/drive/MyDrive/Colab_Notebooks/PRETRAINING/commonsense/commonsense10")
      model.active_adapters = commonsense10
      """

      model.cuda()
      optimizer = AdamW(model.parameters(),
                        lr = 2e-5, # args.learning_rate - default is 5e-5.
                        eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                              )

      total_steps = len(train_dataloader) * epochs
      scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                  num_warmup_steps = 0, # Default value in run_glue.py
                                                  num_training_steps = total_steps)

      # ========================================
      #               Training
      # ========================================
      # Perform one full pass over the training set.
      ft_model=finetune_model(model, train_dataloader)

      # ========================================
      #               Validation
      # ========================================
      # After the completion of training, measure our performance on
      # our validation set.
      validate_predictions(ft_model, validation_dataloader, fold_counter)
      fold_counter += 1

      print('For seed: ',seed_val)
      for fold in results_dict:
        for metric in results_dict[fold]:
          for res in results_dict[fold][metric]:
            print(fold,metric,'=',res)
        print('----------')
    results_per_run.append(results_dict)
  
  with open(os.path.join(args.results_path+'baseline_allbinarized.csv'),'w') as outf: #'_times_negs='+str(times_negs)
      finalp = []
      finalr = []
      finalf1 = []
      for idx,rdict in enumerate(results_per_run):
        avgp=[]
        avgr=[]
        avgf1=[]
        for fold in rdict:
          avgp.append(rdict[fold]['p'])
          avgr.append(rdict[fold]['r'])
          avgf1.append(rdict[fold]['f1'])
        avgp = np.mean(avgp)
        avgr = np.mean(avgr)
        avgf1 = np.mean(avgf1)
        finalp.append(avgp)
        finalr.append(avgr)
        finalf1.append(avgf1)
        outf.write(f'For run {idx+1}\n')
        outf.write(f'---------\n')
        outf.write(f'Precision={avgp}\n')
        outf.write(f'Recall={avgr}\n')
        outf.write(f'F1={avgf1}\n')
        outf.write(f'---------\n\n')
      finalp = np.mean(finalp)
      finalr = np.mean(finalr)
      finalf1 = np.mean(finalf1)
      outf.write(f'Results after avg of all runs for category {cat}\n')
      outf.write(f'Precision={finalp}\n')
      outf.write(f'Recall={finalr}\n')
      outf.write(f'F1={finalf1}\n')


