The flow of the experiments is as follows:

1.
- Pretrain model on selected task (in data folder) --> pretrain_model.py
- Train adapter on selected task (in data folder) --> pretrain_adapter.py

2. 
- Use the finetuned model to re-finetune and test on PCL (we can't upload the data, under request)  --> setting4_binary_+ftmodel.py
- Use the baseline model with the selected adapter to fine-tune and test on PCL - BINARY --> setting4_binary_+adapters.py

3. Because adapter works better than full-finetuned model, we use adapters to fine-tune and test on PCL - MULTILABEL --> setting4_multilabel_+adapters.py

