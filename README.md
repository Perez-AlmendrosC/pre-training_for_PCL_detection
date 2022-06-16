The flow of the experiments is as follows:

1.
- Finetune model on selected task (in data folder) --> This code is in a notebook,I'll upload it.
- Train adapter on selected task (in data folder)

2. 
- Use the finetuned model to re-finetune and test on PCL (we can't upload the data, under request)
- Use the baseline model with the selected adapter to fine-tune and test on PCL - BINARY

3. Because adapter works better than full-finetuned model, we use adapters to fine-tune and test on PCL - MULTILABEL

