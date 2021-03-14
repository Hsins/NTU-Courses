"""
$1 $2 $3 $4
<test.csv path> <prediction file path> <movies.csv path> <users.csv path>
"""
import sys
import csv
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.layers import *

def load_train(train_path):
	train_data=pd.read_csv(train_path,sep=',')
	return train_data

def load_(test_path,movies_path,users_path):
    test_data=pd.read_csv(test_path,sep=',')
    users = pd.read_csv(users_path,sep='::')
    movies=pd.read_csv(movies_path,sep='::')
    return test_data,movies,users

def Bset_model(n_users, n_items, latent_dim = 1024):
    user_input = Input(shape=[1])
    item_input = Input(shape = [1])
    #input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
    #https://faroit.github.io/keras-docs/2.0.8/layers/embeddings/#embedding
    user_vec_dot = Embedding(input_dim = n_users, output_dim= latent_dim, embeddings_initializer="random_normal")(user_input)
    user_vec_dot = Flatten()(user_vec_dot)
    
    item_vec_dot = Embedding(input_dim = n_items+1, output_dim= latent_dim, embeddings_initializer="random_normal")(item_input)
    item_vec_dot = Flatten()(item_vec_dot)
    
    user_vec_Con = Embedding(input_dim = n_users, output_dim= latent_dim)(user_input)
    #user_vec_Con = Embedding(input_dim = n_users, output_dim= latent_dim, embeddings_initializer="random_normal")(user_input)
    user_vec_Con = Flatten()(user_vec_Con)
    #user_vec_Con = Dropout(0.1)(user_vec_Con)


    item_vec_Con = Embedding(input_dim = n_items+1, output_dim= latent_dim)(item_input)
    #item_vec_Con = Embedding(input_dim = n_items, output_dim= latent_dim, embeddings_initializer="random_normal")(item_input)
    item_vec_Con = Flatten()(item_vec_Con)
    #item_vec_Con = Dropout(0.1)(item_vec_Con)

    # user_bias = Embedding(input_dim = n_users, output_dim= 1, embeddings_initializer="zeros")(user_input)
    user_bias = Embedding(input_dim = n_users, output_dim= 1)(user_input)
    user_bias = Flatten()(user_bias)
    
    # item_bias = Embedding(input_dim = n_items+1, output_dim= 1, embeddings_initializer="zeros")(item_input)       
    item_bias = Embedding(input_dim = n_items+1, output_dim= 1)(item_input)

    item_bias = Flatten()(item_bias)
    
    
    r_hot = Dot(axes=1)([user_vec_dot,item_vec_dot])
    r_hot = Add()([r_hot,user_bias, item_bias])

    embedding_Con = Concatenate()([user_vec_Con, item_vec_Con,r_hot])
    hidden_1 = Dense(64,activation = 'linear',kernel_regularizer=l2(5))(embedding_Con)
    hidden_1 = Dropout(0.5)(hidden_1)

    # hidden_2 = Dense(32,activation = 'linear',kernel_regularizer=l2(10))(hidden_1)
    # hidden_2 = Dropout(0.25)(hidden_2)

    pred = Dense(1, activation='linear',kernel_regularizer=l2(25))(hidden_1)

    # outt = Concatenate()([r_hot,pred])
    # outt = Dense(1, activation='linear',kernel_regularizer=l2(1))(outt)


    model = Model([user_input, item_input], pred)
    model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse'])
    return model

#MF model
def get_model(n_users, n_items, latent_dim = 150):
    user_input = Input(shape = [1])
    item_input = Input(shape = [1])
    user_vec = Embedding(n_users,latent_dim,embeddings_initializer = 'random_normal')(user_input)
    user_vec = Dropout(0.1)(user_vec)
    user_vec = Flatten()(user_vec)
    #user_vec = Activation('tanh')(user_vec)

    item_vec = Embedding(n_items,latent_dim,embeddings_initializer = 'random_normal')(item_input)
    item_vec = Dropout(0.1)(item_vec)
    item_vec = Flatten()(item_vec)
    #item_vec = Activation('tanh')(item_vec)

    user_bias = Embedding(n_users,1,embeddings_initializer = 'uniform')(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_items,1,embeddings_initializer = 'uniform')(item_input)
    item_bias = Flatten()(item_bias)

    r_hat = Dot(axes = 1)([user_vec,item_vec])
    r_hat = Add()([r_hat,user_bias,item_bias])
    model = Model([user_input,item_input],r_hat)
    model.compile(loss = 'mse',optimizer = 'adam')
    return(model)


def main(*args):
	test_data,movies,users=load_(args[0][1],args[0][3],args[0][4])
	#test_data,movies,users=load_('./feature/test.csv','./feature/movies.csv','./feature/users.csv')
	#latent_dim = 120
	#n_users= np.max(users['UserID'])
	#n_movies=np.max(movies['movieID'])
	"""
	model = get_model1(n_users,n_movies,latent_dim)
	model.summary()


	ACCearlyStopping=EarlyStopping(monitor='val_loss',patience=50, verbose=0, mode='auto')
	filepath="model1/BOW_weights-{epoch:02d}-{val_acc:.4f}.hdf5"
	CHECKPOINT = ModelCheckpoint("MF1_best_model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')


	train_history = model.fit([X_train['UserID'],X_train['MovieID']],y_train,batch_size=2048,
	    epochs=150,verbose=1,validation_data=([X_valid['UserID'], X_valid['MovieID']], y_valid),
	    callbacks=[ACCearlyStopping,CHECKPOINT])
	"""
	x = test_data['UserID']
	y = test_data['MovieID']

	model = load_model('hw5_MF_model.hdf5')
	prediction = model.predict([x,y])

	answer = []
	for i in range(len(prediction)):
	    answer.append([str(i+1)])
	    if prediction[i][0] <1:
	    	answer[i].append(1)
	    elif prediction[i][0] >5:
	    	answer[i].append(5)
	    else:
	    	answer[i].append(prediction[i][0])

	filename = args[0][2]
	#filename = "MF_Best_prediction.csv"
	text = open(filename, "w+")
	s = csv.writer(text,delimiter=',',lineterminator='\n')
	s.writerow(["TestDataID","Rating"])
	for i in range(len(answer)):
	    s.writerow(answer[i])
	text.close()

	#print('----prediction----')

if __name__ == '__main__':
	main(sys.argv)
	#main()