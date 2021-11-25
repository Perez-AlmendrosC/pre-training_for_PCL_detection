# ethics-pcl

Code needed: 
1. Get sentence embeddings from the aux. tasks. (Aux task ==> data folder, any of the _clean.csv files)
2. Average embeddings for that task data.
3. Get sentence embeddings from main task (DPM - binary).
4. Average embeddings for PCL data.
5. Get the cosine similarity between both embeddings.
(6. Rank the aux.tasks by similarity with PCL task)

### Work done

1. Go to [this notebook](https://colab.research.google.com/drive/1-CEZKfcJ_-LLtYm4SUXJlAX_1_YuvV35?usp=sharing) to embed a dataset into one vector, it will be saved in the colab env, just download when the notebook is done. If you get ouf of memory cuda error, reduce the batch size, use a smaller LM, or run this notebook in flexilog/cluster.
2. Then, upload the vectors to [this notebook](https://colab.research.google.com/drive/1BnuEmlYsUt0pRKsA-TcMEHzxxNlfHWtg?usp=sharing), run the notebook, and you should get, for each of the datasets, their most similar.
