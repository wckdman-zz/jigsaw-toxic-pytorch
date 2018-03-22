This is a part of my solution (top-4%) for **Kaggle Jigsaw Toxic Comment Classification Challenge** - https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge


# Structure

- preprocess.py - creating vocabulary, trimming sentences, creating embedding matrix, converting sentences into sequences
- utils.py - training and evaluation; preparing submission file
- Models.py - code for models.
- DataLoader.py - dataloaders
- train.py - training process, logging, 10-fold crossvalidation

# How to

1. Create **`data/`** and **`predictions/`** directories.
2. Save preprocessed and tokenized train/test text data into separate files. After that run the command below to create an input file with sequences, embedding matrix and vocabulary

```
python3 preprocess.py -train_src data/train.data -train_tgt data/target.data -test_src data/test.data -save_data input.data
```
3. Now you're ready to train NN-models:
```
python3 train.py -name gru_experiment_1 -data input.data -epoch 10 -batch_size 512 -log
```


# Solution overview

On preprocessing stage I extracted a subset of 'toxic' words from vocabulary and replaced rare words with more frequent according Levenshtein distance.
My final model was a weighted average of dozen NN, several linear models and LightGBM. I tried several architectures such as conv1d, BiGRU, BiGRU + conv1d, different pretrained embeddings and different vocabulary and input sizes. My best single model was BiGRU+conv1d+GlobalMaxPooling which gave me 0.9850 AUC on private LB.


