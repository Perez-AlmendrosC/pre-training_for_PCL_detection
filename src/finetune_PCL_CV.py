from simpletransformers.classification import MultiLabelClassificationModel,ClassificationModel
import pandas as pd
import logging
from collections import defaultdict

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold,StratifiedKFold

import pandas as pd
import torch

import numpy as np
import random
import os
import sys


def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(seed)
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)


class CorpusIter:

	def __init__(self,corpusfile):
		self.c = corpusfile
	def __iter__(self):
		for line in open(self.c):
			yield line

def get_pretrained_name(modelpath):
	#'/scratch/c.c1867383/ethics_finetune/cm/bert-base2'
	modelname = modelpath.split('/')[-1]
	pretrain_epochs = ''
	for i in modelname:
		if i.isdigit():
			pretrain_epochs += i
	modelname = modelname.replace(pretrain_epochs, '')
	return modelname,pretrain_epochs

if __name__ == '__main__':

	args = sys.argv[1:]

	if len(args) == 5:

		corpus_path = args[0] #'dontpatronizeme_pcl.txt'
		model_name = args[1] #'bert'
		model_path = args[2] #'models/justice_model/bert/5epochs'
		out_model = args[3] #'models/justice_model/fine_tuned_bert5'
		results_path = args[4] #'models/justice_model/results'

		logging.basicConfig(level=logging.INFO)
		transformers_logger = logging.getLogger("transformers")
		transformers_logger.setLevel(logging.WARNING)

		SEED = 1

		LR = [2e-5, 1e-5] # default 2e-5  2e-3, 2e-7 , 1e-5
		EPOCHS = [2,5,10] # default 2,5,10
		BATCH_SIZE = [8] # default 8,16
		FOLDS = [10]

		rows=[]

		with open(corpus_path) as data:
			for line in data:
				t=line.strip().split('\t')[3].lower()
				l=line.strip().split('\t')[-1]
				if l=='0' or l=='1':
					lbin=0
				else:
					lbin=1
				rows.append(
					{'paragraph':t, 
					'label':lbin}
					)
		print(len(rows))

		down_sample=[]
		negative_examples=[]

		for dictionary in rows:
			if dictionary['label']==0:
				negative_examples.append(dictionary)
			else:
				down_sample.append(dictionary)

		neg_examples=negative_examples[:3000]
		for d in neg_examples:
			down_sample.append(d)

		data_df=pd.DataFrame(down_sample, columns=['paragraph', 'label']) 
		data_df.head()

		# Get the lists of sentences and their labels.
		X = data_df.paragraph.values
		y = data_df.label.values

		# Create or load the classification model
		set_seed(SEED)

		prefinetuned_model, prefinetuned_epochs = get_pretrained_name(model_path)
		
		for folds in FOLDS:
			for epochs in EPOCHS: 
				for batch_size in BATCH_SIZE:
					for lr in LR:
						all_p = []
						all_r = []
						all_f1 = []
						kf = KFold(n_splits=folds, shuffle=True, random_state=1)					
						out_file_name=f'prefm={prefinetuned_model}_prefe={prefinetuned_epochs}_thismodel={model_name}_folds={folds}_epochs={epochs}_batch_size={batch_size}_lr={lr}.txt'
						with open(os.path.join(results_path,out_file_name),'a') as output_file:
							c=1
							config_str = f'''
								downsample_pcl_finetuned with {prefinetuned_model} on {prefinetuned_epochs} epochs 
								folds = {folds}
								epochs = {epochs}
								batch size = {batch_size} 
								lr = {lr}
								-------------------------------------------------------------------------------
								fold_numb = {c}
								-------------------------------------------------------------------------------\n
								'''
							print(config_str)
							for train_index, val_index in kf.split(data_df):
								model = ClassificationModel(model_name, model_path, num_labels=2, 
															args={'reprocess_input_data': True, 
																'overwrite_output_dir': True, 
																'num_train_epochs': epochs, 
																'cache_dir':out_model, 
																'output_dir':out_model, 
																'n_gpu':2, 
																'silent':False, 
																'train_batch_size':batch_size, 
																'learning_rate': lr
																}) 
								print('Fold number ',c,':', '\n')
								print('Training...')
								train_df = data_df.iloc[train_index]
								val_df = data_df.iloc[val_index]
								model.train_model(train_df, output_dir=out_model)
								print('Done training, getting predictions')

								#result, model_outputs, wrong_predictions = model.eval_model(val_df, acc=accuracy_score)
								#print('Accuracy = ', result['acc'])

								pred, raw_outputs = model.predict(val_df.paragraph.values.tolist())

								gold = val_df.label.values
								res = classification_report(gold, pred, digits=4)#, output_dict = True)

								print('for fold ',c,' the classification report is: ','\n', res)
								
								print('\n')
								print('------------------------------------------')
								print('\n')
							
								output_file.write('Classification report:')
								output_file.write(config_str)
								output_file.write(res)

								p = precision_score(gold, pred)
								r = recall_score(gold, pred)
								f1 = f1_score(gold, pred)

								all_p.append(p)
								all_r.append(r)
								all_f1.append(f1)

								c+=1
						
						avg_p = np.mean(all_p)
						avg_r = np.mean(all_r)
						avg_f1 = np.mean(all_f1)
							
						print('=== Averaged results over {folds} folds ===')
						print('P: ',avg_p)
						print('R: ',avg_r)
						print('F1: ',avg_f1)


