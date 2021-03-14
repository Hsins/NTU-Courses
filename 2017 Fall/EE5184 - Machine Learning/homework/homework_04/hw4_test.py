#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:06:43 2017
@author: adahsieh

https://github.com/RaRe-Technologies/gensim/tree/develop/docs/notebooks

"""
#import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.models import Model, load_model

from keras import regularizers
from keras.preprocessing.text import Tokenizer,text_to_word_sequence

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam, Adadelta

from keras.layers.recurrent import LSTM

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence
import sys
import re
import itertools
from gensim.models import word2vec
import logging

import csv

from sklearn.feature_extraction.text import CountVectorizer
from six import iteritems
from gensim import corpora
from itertools import chain
from collections import Counter


MAX_SEQUENCE_LENGTH=40
EMBEDDING_DIM =256

MAX_NB_WORDS = 20000

VALIDATION_SPLIT = 0.1
BATCH_SIZE = 1024
NUM_EPOCH = 20

path = './hw_4_3.txt'


#--------------------------------------load data
def load_train(readnpy=False):
	if readnpy:
		label_y = np.load('./feature/label_y.npy')
		label_x = np.load('./feature/label_x.npy')
		nolabel_x = np.load('./feature/nolabel_x.npy')
	else:
		with open('./feature//training_label.txt',"U", encoding = 'utf-8-sig') as f:
			label_y = []
			label_x = []
			nolabel_x = []
			for l in f:
				label_y.append(l.strip().split("+++$+++")[0])
				label_x.append(l.strip().split("+++$+++")[1])
			np.save('./feature/label_y.npy', label_y)
			np.save('./feature/label_x.npy', label_x)
			'''
		with open('./training_nolabel.txt') as f:
			for l in f:
				test_text.append(l.strip())
				np.save('./feature/train_text.npy', train_text)
			'''
		nolabel_x = [line.strip() for line in open('./feature/training_nolabel.txt',"U", encoding = 'utf-8-sig')]
		np.save('./feature/nolabel_x.npy',nolabel_x)
	return label_y,label_x,nolabel_x

#deal with test data
def load_test(readnpy=False):
	if readnpy:
		test_text=np.load('./feature/test_text.npy')
	else:
		test_text = [line.strip().split(',',1)[1] for line in open(
			'./feature/testing_data.txt', "r", encoding='utf-8-sig')][1::]
		np.save('./feature/test_text.npy', test_text)
	return test_text

def load_testdata(testing):
	test_text = [line.strip().split(',',1)[1] for line in open(
		testing, "r", encoding='utf-8-sig')][1::]
	return test_text

def load_traindata(training_label,training_nolabel):
	with open(training_label,"U", encoding = 'utf-8-sig') as f:
			label_y = []
			label_x = []
			nolabel_x = []
			for l in f:
				label_y.append(l.strip().split("+++$+++")[0])
				label_x.append(l.strip().split("+++$+++")[1])
	nolabel_x = [line.strip() for line in open('./feature/training_nolabel.txt',"U", encoding = 'utf-8-sig')]
	return label_y,label_x,nolabel_x

def textprcoessing(text):
	text = pd.DataFrame(text)
	text = text[0].apply(lambda x:''.join([i for i in x if i in 'abcdefghijklmnopqrstuvwxyz ']))
	return text

def getmodel_word_2vec_embedding(sentence=None,load=False):
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 
	if load:
		model=word2vec.Word2Vec.load('model_word2vec')
	else:
		sentence=[s.split() for s in sentence]
		"""
		# remove common words and tokenize
		stoplist = set(stopwords.words("english"))  
		#stoplist = set('for a of the and to in'.split())
		sentence = [[word for word in s.lower().split() if word not in stoplist]
				for s in sentence]

		print(sentence[:5])
		
		# remove words that appear only once
		frequency = defaultdict(int)
		for text in texts:
		for token in text:
			frequency[token] += 1
		
		texts = [[token for token in text if frequency[token] > 1] for text in texts]
		pprint(texts[:5])
		"""
		model = word2vec.Word2Vec(sentence, min_count=5,workers=8,size=256,iter=7,sg=1)
		model.save('./model1/model_word2vec')
	return model


def process_for_data(model,train=None,test=None):
	word2idx = {"_PAD": 0}
	vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
	embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
	
	for i in range(len(vocab_list)):
		word = vocab_list[i][0]
		word2idx[word] = i + 1
		embeddings_matrix[i + 1] = vocab_list[i][1]
	return word2idx,embeddings_matrix

def get_data_index(word_index,train,test,load=False):   
	if load:
		train_index= np.load('./feature/train_index.npy')
		test_index=np.load('./feature/test_index.npy')
	else:
		train=[s.split() for s in train]
		test=[s.split() for s in test]
		
		train_index=np.zeros((len(train), MAX_SEQUENCE_LENGTH), dtype='float32')
		test_index=np.zeros((len(test), MAX_SEQUENCE_LENGTH), dtype='float32')
		for i in range(0,len(train)):
			for j in range(0,len(train[i])):
				try:
					train_index[i][j]=word_index[train[i][j]]
				except KeyError:
					train_index[i][j]=0
		for i in range(0,len(test)):
			for j in range(0,len(test[i])):
				try:
					test_index[i][j]=word_index[test[i][j]]
				except KeyError:
					test_index[i][j]=0
		np.save('./feature/train_index.npy', train_index)
		np.save('./feature/test_index.npy', test_index)
	return train_index,test_index

def get_train_index(word_index,train):
	train=[s.split() for s in train]
		
	train_index=np.zeros((len(train), MAX_SEQUENCE_LENGTH), dtype='float32')
		
	for i in range(0,len(train)):
		for j in range(0,len(train[i])):
			try:
				train_index[i][j]=word_index[train[i][j]]
			except KeyError:
				train_index[i][j]=0
	return train_index

def text_process(sentence):
	sentence=[s.split() for s in sentence]
	"""
	# remove common words and tokenize
	stoplist = set(stopwords.words("english"))  
	#stoplist = set('for a of the and to in'.split())
	sentence = [[word for word in s.lower().split() if word not in stoplist]
				for s in sentence]

	print(sentence[:5])
		
	# remove words that appear only once
	frequency = defaultdict(int)
	for text in texts:
		for token in text:
			frequency[token] += 1
		
	texts = [[token for token in text if frequency[token] > 1] for text in texts]
	pprint(texts[:5])
	"""
	return sentence

def transform_bow_matrix(word_2idx,transform_idx):
	corp=[]
	index=0
	for i in transform_idx:
		corp.append(dict(Counter(transform_idx[index])))
		index += 1
	text_matrix = []
	for i in range(0,len(transform_idx)):
		b = [0]*len(word_2idx)
		k = list(corp[i].keys())
		for p in k:
			b[int(p)] = corp[i][p]
		text_matrix.append(b)
	return text_matrix

def bagofwords(text,label_x,label_y,test_text):

	print("----------------------------------Start BOW------------------------------------------------")
	#text = text_process(text)

	cv=CountVectorizer()
	text_transform = cv.fit_transform(text)
	#print(cv.get_feature_names())
	#text_transform.toarray()

	label_x = list(chain.from_iterable(label_x))
	label_x = cv.transform(label_x)
	#label_x.toarray()
	type(label_x)
	#print(label_x.toarray().shape)

	test_text = list(chain.from_iterable(test_text))
	test_text = cv.transform(test_text)
	#test_text.toarray()

	#print(test_text.toarray().shape)

	BOW_model = Sequential()
	BOW_model.add(Dense(input_dim=label_x.shape[1],units =512,
		activation='relu',kernel_regularizer=regularizers.l2(0.001)))
	BOW_model.add(Dense(256, activation='relu'))
	BOW_model.add(Dropout(0.4))
	BOW_model.add(Dense(units=1, activation='relu'))
	BOW_model.compile(loss='binary_crossentropy',
	              optimizer=Adam(lr=1e-3), metrics=['accuracy'])
	BOW_model.summary()

	ACCearlyStopping=EarlyStopping(monitor='val_acc',patience=5, verbose=0, mode='auto')
	filepath="model1/BOW_weights-{epoch:02d}-{val_acc:.4f}.hdf5"
	CHECKPOINT = ModelCheckpoint("model1/semi_model.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	
	BOW_train_history =BOW_model.fit(label_x,label_y,batch_size=BATCH_SIZE,verbose=2,
		epochs=NUM_EPOCH,
		callbacks=[CHECKPOINT,ACCearlyStopping],validation_split=VALIDATION_SPLIT)
	
	#show_train_history(BOW_train_history, 'acc', 'val_acc')
	#show_train_history(BOW_train_history, 'loss', 'val_loss')

	BOW_prediction = BOW_model.predict_classes(test_text, batch_size=20000)
	print("end predict")

	savetocsv=input('write the prediction to csv:(y/n)').lower().strip()

	if savetocsv=="y":
		answer = []
		for i in range(len(BOW_prediction)):
			answer.append([str(i)])
			answer[i].append(BOW_prediction[i][0])

		#filename = args[0][6]
		filename = "result/BOW_prediction.csv"

		text = open(filename, "w+")
		s = csv.writer(text,delimiter=',',lineterminator='\n')
		s.writerow(["id","label"])
		for i in range(len(answer)):
			s.writerow(answer[i])
		text.close()
	else:
		print("end")

	return BOW_model

def bag_of_words(text):
	text = text_process(text)
	cv=CountVectorizer()
	text_transform = cv.fit_transform(text)
	print(cv.get_feature_names())
	text_transform.toarray()
	return cv,text_transform.toarray()

def transform_bow(cv,new_text):
	new_text = list(chain.from_iterable(new_text))
	new_text = cv.transform(new_text)
	new_text.toarray()
	return new_text.toarray()

def create_dic(text):
	dictionary = corpora.Dictionary(text)
	print(dictionary)
	# remove stop words and words that appear only once
	stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
	once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
	# remove stop words and words that appear only once
	dictionary.filter_tokens(stop_ids + once_ids)
	print(dictionary)

	"""
tags = [
  "python, tools",
  "linux, tools, ubuntu",
  "distributed systems, linux, networking, tools",]

list_of_new_documents = [
  ["python, chicken"],
  ["linux, cow, ubuntu"],
  ["machine learning, bird, fish, pig"]]

vect = CountVectorizer()
tags = vect.fit_transform(tags)
tags.toarray()

# vocabulary learned by CountVectorizer (vect)
print(vect.vocabulary_)
# to use `transform`, `list_of_new_documents` should be a list of strings 
# `itertools.chain` flattens shallow lists more efficiently than list comprehensions

new_docs = list(chain.from_iterable(list_of_new_documents)
new_docs = vect.transform(new_docs)
new_docs.toarray()

	"""
def BOW(label_x,label_y,nolabel_x,test_x):

	print("----------------------------------Start BOW------------------------------------------------")

	documents = np.concatenate([label_x, nolabel_x, test_x])

	tokenizer = Tokenizer(nb_words=10000)
	tokenizer.fit_on_texts(documents)
	#print(tokenizer.word_index)
	#tokenizer.texts_to_sequences(label_x)

	x_train_BoW = tokenizer.texts_to_matrix(label_x,mode='count')
	x_test_BoW = tokenizer.texts_to_matrix(test_x,mode='count')

	BOW_model = Sequential()
	BOW_model.add(Dense(input_dim=x_train_BoW.shape[1],units =512,
		activation='relu',kernel_regularizer=regularizers.l2(0.001)))
	BOW_model.add(Dense(256, activation='relu'))
	BOW_model.add(Dropout(0.4))
	BOW_model.add(Dense(units=1, activation='relu'))
	BOW_model.compile(loss='binary_crossentropy',
	              optimizer=Adam(lr=1e-3), metrics=['accuracy'])
	BOW_model.summary()

	ACCearlyStopping=EarlyStopping(monitor='val_acc',patience=5, verbose=0, mode='auto')
	filepath="model1/BOW_weights-{epoch:02d}-{val_acc:.4f}.hdf5"
	CHECKPOINT = ModelCheckpoint("model1/BOW_model.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	
	BOW_train_history =BOW_model.fit(x_train_BoW,label_y,batch_size=BATCH_SIZE,verbose=2,
		epochs=NUM_EPOCH,
		callbacks=[CHECKPOINT,ACCearlyStopping],validation_split=VALIDATION_SPLIT)
	
	#show_train_history(BOW_train_history, 'acc', 'val_acc')
	#show_train_history(BOW_train_history, 'loss', 'val_loss')

	BOW_prediction = BOW_model.predict_classes(x_test_BoW, batch_size=20000)
	print("end predict")

	savetocsv=input('write the prediction to csv:(y/n)').lower().strip()

	if savetocsv=="y":
		answer = []
		for i in range(len(BOW_prediction)):
			answer.append([str(i)])
			answer[i].append(BOW_prediction[i][0])

		#filename = args[0][6]
		filename = "result/BOW_prediction.csv"

		text = open(filename, "w+")
		s = csv.writer(text,delimiter=',',lineterminator='\n')
		s.writerow(["id","label"])
		for i in range(len(answer)):
			s.writerow(answer[i])
		text.close()
	else:
		print("end")
	return BOW_model,tokenizer

"""
def for_hw4_3():
	
	for_text = [line.strip().split(',', 1)[1] for line in open(
	        path, "r", encoding='utf-8-sig')][1::]

	for_list = [text_to_word_sequence(s, filters='', lower=True, split=" ") for s in for_text]
	for_train = np.array( [index_array(x, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM) for x in for_list])
	
	RNN_prediction = model.predict(for_train)

	for_train_BoW = tokenizer.texts_to_matrix(for_text,mode='count')
	BOW_prediction = BOW_model.predict(toy_train_BoW)
"""
#--------------------------------------display

"""
def show_train_history(train_history,train,validation):
	plt.plot(train_history.history[train])
	plt.plot(train_history.history[validation])
	plt.title('Train History')
	plt.ylabel(train)
	plt.xlabel('Epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()
"""
SentimentDict={1:'正面的',0:'負面的'}
def display_test_Sentiment(i):
	print(test_text[i])
	print('標籤label:',SentimentDict[y_test[i]],'預測結果:',SentimentDict[predict_classes[i]])

#--------------------------------------display

def build_model(embeddings_matrix):
	model = Sequential()
	embedding_layer = Embedding(input_dim=len(embeddings_matrix),
							output_dim=EMBEDDING_DIM,
							input_length=MAX_SEQUENCE_LENGTH,
							weights=[embeddings_matrix],
							trainable=False)
	model.add(embedding_layer)
	model.add(Dropout(0.1))
	model.add(LSTM(256))
	model.add(Dropout(0.1))
	#model.add(Bidirectional(LSTM(64)))
	#model.add(SimpleRNN(units=256))
	#model.add(GRU(units=256))
	#model.add(Dense(units=256,kernel_regularizer=regularizers.l2(0.1),activation='relu'))
	#model.add(Dropout(0.4))
	model.add(Dense(units=128, activation='relu' ))
	model.add(Dense(units=1, activation='sigmoid'))
	# opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	# opt = Adam(lr=1e-3)
	# opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	model.compile(loss = 'binary_crossentropy',optimizer=Adam(), metrics=['accuracy'])
	return model

def prediction(test_X,model_path='best_model.h5'):
	model = load_model(model_path)
	pred = [1 if x[0] >= 0.5 else 0 for x in model.predict(test_X ,batch_size=512, verbose=1)]
	text = open('predict.csv', "w+")
	s = csv.writer(text, delimiter=',', lineterminator='\n')
	s.writerow(["id", "label"])
	[s.writerow([i,pred[i]]) for i in range(len(pred))]
	text.close()


def main(*args):
	#label_y,label_x,nolabel_x=load_traindata(args[0][1],args[0][2])
	#label_y,label_x,nolabel_x=load_train(True)
	#test_x=load_test(True)
	test_x = load_testdata(args[0][1])

	#text = np.concatenate([label_x,test_x,nolabel_x])
	#text = np.concatenate([label_x,nolabel_x])

	#nolabel_x=textprcoessing(nolabel_x)	
	#label_x=textprcoessing(label_x)
	#text=textprcoessing(text)
	
	test_x=textprcoessing(test_x)
	
	#BOW_model,BOW_token = BOW(label_x,label_y,nolabel_x,test_x)
	#BOW_model=bagofwords(text,label_x,label_y,test_x)

	
	#----bag of words
	#text_transform,label_x_transform,test_x_transform = bagofwords(text,label_x,test_x)

	word_2vec=getmodel_word_2vec_embedding(text_x,True)
	#print(word_2vec.most_similar(['like']))
	#print(word_2vec.most_similar(['shut']))
	word_2idx,embeddings_matrix=process_for_data(word_2vec)

	#train_idx,test_idx=get_data_index(word_2idx,label_x,test_x)
	test_idx = get_train_index(word_2idx,test_x)


	# print(len(train_idx))
	# print(len(test_idx))

	#----bag of words
	#train_matrix = transform_bow_matrix(word_2idx,train_idx)
	#test_matrix = transform_bow_matrix(word_2idx,test_idx)
	#----bag of words
	"""
	model = build_model(embeddings_matrix)
	model.summary()
	ACCearlyStopping=EarlyStopping(monitor='val_acc',patience=10, verbose=0, mode='auto')
	filepath="weights-{epoch:02d}-{val_acc:.4f}.hdf5"
	CHECKPOINT = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	
	train_history =model.fit(train_idx,label_y,batch_size=BATCH_SIZE,verbose=2,
		epochs=NUM_EPOCH,
		callbacks=[CHECKPOINT,ACCearlyStopping],validation_split=VALIDATION_SPLIT)
	
	show_train_history(train_history, 'acc', 'val_acc')
	show_train_history(train_history, 'loss', 'val_loss')
	"""

	model_path="hw4_best_model.hdf5"
	model = load_model(model_path)
	
	prediction = model.predict_classes(test_idx,batch_size=20000)
	print("end predict")

	answer = []
	for i in range(len(prediction)):
		answer.append([str(i)])
		answer[i].append(prediction[i][0])

	filename = args[0][2]
	#filename = "result/prediction_.csv"
	text = open(filename, "w+")
	s = csv.writer(text,delimiter=',',lineterminator='\n')
	s.writerow(["id","label"])
	for i in range(len(answer)):
		s.writerow(answer[i])
	text.close()

	"""
	savetocsv=input('write the prediction to csv:(y/n)').lower().strip()

	if savetocsv=="y":
		answer = []
		for i in range(len(prediction)):
			answer.append([str(i)])
			answer[i].append(prediction[i][0])

		filename = args[0][2]
		#filename = "result/prediction_.csv"

		text = open(filename, "w+")
		s = csv.writer(text,delimiter=',',lineterminator='\n')
		s.writerow(["id","label"])
		for i in range(len(answer)):
			s.writerow(answer[i])
		text.close()
	else:
		print("end")
	"""
if __name__ == '__main__':
	main(sys.argv)



