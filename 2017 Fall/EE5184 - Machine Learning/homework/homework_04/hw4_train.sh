#!/bin/bash 
wget -O hw4_model.hdf5 https://www.dropbox.com/s/c68r5yxzhc7mkhm/hw4_best_model.hdf5?dl=1
wget -O model_word2vec.wv.syn0.npy https://www.dropbox.com/s/n4yhsdw2gogaax3/model_word2vec.wv.syn0.npy?dl=1
wget -O model_word2vec.syn1neg.npy https://www.dropbox.com/s/amdv7ew95jm8duf/model_word2vec.syn1neg.npy?dl=1
wget -O model_word2vec https://www.dropbox.com/s/cc343ts808zkueh/model_word2vec?dl=1
python hw4_train.py $1 $2