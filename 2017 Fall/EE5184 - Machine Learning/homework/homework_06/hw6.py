import numpy as np
from math import floor
import random

from keras.models import Model
from keras.layers import Dense, Input

from keras.models import load_model
import sys
from sklearn.cluster import KMeans
import csv

autoencoder_model_path = 'hw6_autoencoder.h5'
encoder_model_path = 'hw6_encoder.h5'
random.seed(24)

EPOCHS = 100
BATCH_SIZE = 5000
VALID_SPLIT = 0.1

def _shuffle(X):
	randomize = np.arange(len(X))
	np.random.shuffle(randomize)
	return (X[randomize])

def split_valid_set(X_all, percentage):
	all_data_size = len(X_all)
	valid_data_size = int(floor(all_data_size * percentage))
	
	X_all= _shuffle(X_all)
	
	X_valid= X_all[0:valid_data_size]
	X_train= X_all[valid_data_size:]
	
	return X_train, X_valid

def model(x_train,encoding_dim=64):
	# this is our input placeholder
	input_img = Input(shape=(784,))

	# encoder layers
	encoded = Dense(512, activation='relu')(input_img)
	encoded = Dense(256, activation='relu')(encoded)
	encoded = Dense(128, activation='relu')(encoded)
	#encoded = Dense(64, activation='relu')(encoded)
	encoder_output = Dense(encoding_dim)(encoded)

	# decoder layers
	decoded = Dense(encoding_dim, activation='relu')(encoder_output)
	#decoded = Dense(128, activation='relu')(decoded)
	decoded = Dense(256, activation='relu')(decoded)
	decoded = Dense(512, activation='relu')(decoded)
	decoded = Dense(784, activation='tanh')(decoded)

	# construct the autoencoder model
	autoencoder = Model(input=input_img, output=decoded)
	# construct the encoder model for plotting
	encoder = Model(input=input_img, output=encoder_output)
	# compile autoencoder
	autoencoder.compile(optimizer='adam', loss='mse')

	autoencoder.summary()
	encoder.summary()
	# training
	autoencoder.fit(x_train, x_train,
					epochs=EPOCHS,
					batch_size=BATCH_SIZE,
					shuffle=True,validation_split=VALID_SPLIT)
	return autoencoder,encoder

def load_test(testfile='test_case.csv'):
	test = []
	with open(testfile) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:  
			test.append([row['image1_index'],row['image2_index']])
	return test

"""
def plot(x_test,pre,n=10):
	import matplotlib.pyplot as plt
	# n = how many digits we will display
	plt.figure(figsize=(20, 4))
	for i in range(n):
		# display original
		ax = plt.subplot(2, n, i + 1)
		plt.imshow(x_test[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# display reconstruction
		ax = plt.subplot(2, n, i + 1 + n)
		plt.imshow(pre[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()
"""

#def main():
def main(*args):
	#imagedata = np.load('image.npy')
	#args[0][1]
	imagedata = np.load(args[0][1])

	#data pre-processing
	imagedata = imagedata.astype('float32') / 255. - 0.5
	x_train,x_test = split_valid_set(imagedata,0.1)

	#test = load_test('test_case.csv')
	#args[0][2]
	test = load_test(args[0][2])

	"""
	#use PCA -> Dimension Reduction
	pca = PCA(n_components=64)
	pca.fit(imagedata)
	pca_extraction = pca.transform(imagedata)
	print(pca_extraction.shape)

	kmeans = KMeans(n_clusters=2, random_state=0).fit(pca_extraction)
	y_kmeans = kmeans.predict(pca_extraction)

	"""

	"""
	encoding_dim = 64
	autoencoder,encoder=model(x_train,encoding_dim)

	autoencoder.save('autoencoder.hdf5')
	encoder.save('encoder.hdf5')
	"""
	autoencoder = load_model(autoencoder_model_path)
	encoder = load_model(encoder_model_path)

	encoded_imgs = encoder.predict(imagedata)
	pre = autoencoder.predict(x_test)
	#autoencoder.summary()

	#print(encoded_imgs.shape)
	#plot(x_test,pre,10)

	kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)
	y_kmeans = kmeans.predict(encoded_imgs)

	answer = []
	for index in range(len(test)):
		answer.append([str(index)])
		image1_result = y_kmeans[int(test[index][0])]
		image2_result = y_kmeans[int(test[index][1])]
		if image1_result == image2_result:
			answer[index].append(1)
		else:
			answer[index].append(0)

	#filename = "result.csv"
	filename = args[0][3]

	text = open(filename, "w+")
	s = csv.writer(text,delimiter=',',lineterminator='\n')
	s.writerow(["ID","Ans"])
	for i in range(len(answer)):
		s.writerow(answer[i])
	text.close()

if __name__ == '__main__':
	main(sys.argv)
	#main()