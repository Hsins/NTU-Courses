
import pandas as pd
from PIL import Image
import numpy as np
import keras
from math import floor
import itertools
import random
from random import shuffle

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta

from sklearn.metrics import confusion_matrix
import csv
from keras.utils import np_utils
import matplotlib.pyplot as plt

import time
import os

os.system('echo $CUDA_VISIBLE_DEVICES')
PATIENCE = 5 # The parameter is used for early stopping

def load(readnpy=True):
    if readnpy:
        y = np.load('./feature/label.npy')
        X = np.load('./feature/feature.npy')
        X_test = np.load('./feature/X_test.npy')
        
        X_train = np.load('./feature/X_train.npy')
        X_valid = np.load('./feature/X_valid.npy')
        y_train = np.load('./feature/y_train.npy')
        y_valid = np.load('./feature/y_valid.npy')        
    else :
        df_train = pd.read_csv('./feature/train.csv')
        
        '''train,valid data'''
        y = df_train['label'].as_matrix()
        print(df_train.groupby('label').count())
        y = y.reshape(len(y),1)
        
        X = df_train['feature'].as_matrix()
        X = np.array([np.array([*map(int, x.split())]).reshape(48,48) for x in X])
        np.save('./feature/label.npy', y)
        np.save('./feature/feature.npy',X)
        
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=0)  
        
        np.save('./feature/X_train.npy',X_train)
        np.save('./feature/X_valid.npy',X_valid)
        np.save('./feature/y_train.npy',y_train)
        np.save('./feature/y_valid.npy',y_valid)
        
        '''testing data'''
        df_test = pd.read_csv('./feature/test.csv')
        X_test = df_test['feature'].as_matrix()
        X_test = np.array([np.array([*map(int, x.split())]).reshape(48,48) for x in X_test])
        np.save('./feature/X_test.npy',X_test)
        
    return X,y,X_test,X_train,X_valid,y_train,y_valid


label_dict={0:"pissed off",1:"disgust",2:"fear",3:"happy",4:"sad",
            5:"surprised",6:"neutral"}

def plotconfusionmatrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_images_labels_prediction(images,labels,prediction,
                                  idx,num=20):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx],cmap='binary')
                
        title=str(i)+','+label_dict[labels[i][0]]
        if len(prediction)>0:
            title+='=>'+label_dict[prediction[i]]
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()

def normalize(X):
    X = X.astype('float32')
    X /=255
    X = X.reshape(len(X),48,48,1)
    return X

def OneHotEncode(y):
    #轉換label 為OneHot Encoding
    y = np_utils.to_categorical(y)
    #y = pd.get_dummies(y).values
    return y

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))
    
    X_all, Y_all = _shuffle(X_all, Y_all)
    
    X_valid, Y_valid = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]
    
    return X_train, Y_train, X_valid, Y_valid

def valid(X_all, Y_all):
    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
    return X_train, Y_train, X_valid, Y_valid

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def prediction(model,X_test):
    prediction=model.predict_classes(X_test)
    print(prediction[:10])

#confusion matrix
def confusionmatrix(y_test,prediction):
    print(label_dict)
    print(y_test.shape)
    print(prediction.shape)
    pd.crosstab(y_test.reshape(-1),prediction,
            rownames=['label'],colnames=['predict'])

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
    X_all,Y_all,X_test,X_train,X_valid,Y_train,Y_valid = load(True)
    
    #plot_images_labels_prediction(X_all,Y_all,[],0)
    
    X_all = normalize(X_all)
    X_test = normalize(X_test)
    Y_all = OneHotEncode(Y_all)
    
    X_train = normalize(X_train)
    X_valid = normalize(X_valid)
    Y_train = OneHotEncode(Y_train)
    #Y_valid = OneHotEncode(Y_valid)

    model = load_model("BEST_weights-improvement-75-0.686.hdf5")
    print(model.summary())    

    prob= model.predict(X_valid)
    prediction=prob.argmax(axis=-1)
    prediction=prediction.reshape(len(prediction),1)

    #confusionmatrix(Y_valid,prediction)    
    conf_mat = confusion_matrix(Y_valid,prediction)
    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
    plt.show()

    #plot_images_labels_prediction(X_test,y_test,prediction,0,10)
    #Predicted_Probability=model.predict(x_img_test_normalize)
    
    #show_Predicted_Probability(y_label_test,prediction,x_img_test,Predicted_Probability,0)
    #show_Predicted_Probability(y_label_test,prediction,x_img_test,Predicted_Probability,3)

if __name__ == '__main__':
    main()

