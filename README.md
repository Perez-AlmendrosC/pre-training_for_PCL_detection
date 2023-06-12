
This is the repository for the paper **[Pre-Training Language Models for Identifying Patronizing and Condescending Language: An Analysis]([url](https://aclanthology.org/2022.lrec-1.415.pdf))**, by Carla Perez Almendros, Luis Espinosa Anke, Steven Schockaert.

In this work, we select 10 auxiliary tasks related to Patronizing and Condescending Language (PCL) with the objective of testing if a langauge model (LM) perviously trained on these tasks would perform better on PCL detection and categorization. In order to test this hypothesis, we experiment with two pretraining strategies, namely full-finetuning of a pre-trained LM and the use of adapters.  

### Auxiliary tasks

 The data related to the auxiliary tasks is contained in the folder \data and corresponds to:
- Commonsense Morality
- Social Justice
- Deontology
- Stereoset
- Offensive Language
- Hate Speech 
- Democrats vs Republicans tweets
- Hyperpartisan News detection
- Irony detection
- Sentiment Analysis

For more information about the datasets, please see [the aforementioned paper]([url](https://aclanthology.org/2022.lrec-1.415.pdf)).

### Experiments and code

The flow of the experiments is as follows:

1. Pre-training
- Pretrain model on selected task (in data folder) --> pretrain_model.py
- Train adapter on selected task (in data folder) --> pretrain_adapter.py

2. Use either the model finetuned on the auxiliary task or add the pre-trained adapter to a pre-trained language model to fine-tune and test on PCL detection. Note that PCL data is only available under request (see https://github.com/Perez-AlmendrosC/dontpatronizeme). 
- Use the file binary_dpm.py or multilabel_dpm.py to re-finetune and test the new models on PCL detection (binary) and categorization (multilabel), where you can either add the pre-trained adapter or inizialize the model from its pre-finetuned version on the selected auxiliary task.


### Citation
If you use our paper or our code, please cite our work as follows: 

```
@inproceedings{perez-almendros-etal-2022-pre,
    title = "Pre-Training Language Models for Identifying Patronizing and Condescending Language: An Analysis",
    author = "Perez Almendros, Carla  and
      Espinosa Anke, Luis  and
      Schockaert, Steven",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.415",
    pages = "3902--3911",
}
```







