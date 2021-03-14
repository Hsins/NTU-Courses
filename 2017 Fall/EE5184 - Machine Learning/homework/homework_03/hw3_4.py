import os
import sys
import argparse
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from termcolor import colored,cprint
import numpy as np
from utils import * 
import pandas as pd
from keras.utils import np_utils

from sklearn import preprocessing

# Saliency map
# https://github.com/experiencor/deep-viz-keras/blob/master/saliency.py
from keras.layers import Input, Conv2DTranspose
from keras.models import Model
from keras.initializers import Ones, Zeros

def normalize(x):
   # utility function to normalize a tensor by its L2 norm
   return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def normal(X):
    X = X.astype('float32')
    X /=255
    X = X.reshape(len(X),48,48,1)
    return X

def OneHotEncode(y):
    #轉換label 為OneHot Encoding
    y = np_utils.to_categorical(y)
    #y = pd.get_dummies(y).values
    return y

class SaliencyMask(object):
    def __init__(self, model, output_index=0):
        pass

    def get_mask(self, input_image):
        pass

    def get_smoothed_mask(self, input_image, stdev_spread=.2, nsamples=50):
        stdev = stdev_spread * (np.max(input_image) - np.min(input_image))

        total_gradients = np.zeros_like(input_image, dtype = np.float64)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, input_image.shape)
            x_value_plus_noise = input_image + noise

            total_gradients += self.get_mask(x_value_plus_noise)

        return total_gradients / nsamples

class GradientSaliency(SaliencyMask):

    def __init__(self, model, output_index = 0):
        # Define the function to compute the gradient
        input_tensors = [model.input]
        print(model.output[0][0])
        print(model.total_loss)
        gradients = model.optimizer.get_gradients(model.output[0][output_index], model.input)
        self.compute_gradients = K.function(inputs = input_tensors, outputs = gradients)

    def get_mask(self, input_image):
        # Execute the function to compute the gradient
        x_value = np.expand_dims(input_image, axis=0)
        gradients = self.compute_gradients([x_value])[0][0]

        return gradients

# https://github.com/experiencor/deep-viz-keras/blob/master/visual_backprop.py
class VisualBackprop(SaliencyMask):
    def __init__(self, model, output_index = 0):
        inps = [model.input]           # input placeholder
        outs = [layer.output for layer in model.layers]    # all layer outputs
        self.forward_pass = K.function(inps, outs)         # evaluation function
        
        self.model = model

    def get_mask(self, input_image):
        x_value = np.expand_dims(input_image, axis=0)
        
        visual_bpr = None
        layer_outs = self.forward_pass([x_value, 0])

        for i in range(len(self.model.layers) - 1, -1, -1):
            if 'Conv2D' in str(type(self.model.layers[i])):
                layer = np.mean(layer_outs[i], axis = 3, keepdims = True)
                layer = layer - np.min(layer)
                layer = layer / (np.max(layer) - np.min(layer) + 1e-6)

                if visual_bpr is not None:
                    if visual_bpr.shape != layer.shape:
                        visual_bpr = self._deconv(visual_bpr)
                    visual_bpr = visual_bpr * layer
                else:
                    visual_bpr = layer

        return visual_bpr[0]
    
    def _deconv(self, feature_map):
        x = Input(shape = (None, None, 1))
        y = Conv2DTranspose(filters = 1, 
                            kernel_size = (3, 3), 
                            strides = (2, 2), 
                            padding = 'same', 
                            kernel_initializer = Ones(), 
                            bias_initializer = Zeros())(x)

        deconv_model = Model(inputs=[x], outputs=[y])

        inps = [deconv_model.input]   # input placeholder                                
        outs = [deconv_model.layers[-1].output]           # output placeholder
        deconv_func = K.function(inps, outs)              # evaluation function
        
        return deconv_func([feature_map, 0])[0]

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_ascent(input_image_data,iter_func):
    # step size for gradient ascent
    step = 5
    #img_asc = np.array(img)
    img_asc = np.random.random((1, 48, 48, 1))
    img_asc = (img_asc - 0.5) * 20 + 48
        
        # run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iter_func([img_asc])
        img_asc += grads_value * step
    img_asc = img_asc[0]
    #img_ascs.append(deprocess_image(img_asc).reshape((48, 48)))        
    return img_asc

def vis_img_in_filter(img,layer_dict,model,
                      layer_name = 'conv2d_2'):
    layer_output = layer_dict[layer_name].output
    img_ascs = list()
    for filter_index in range(layer_output.shape[3]):
        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        loss = K.mean(layer_output[:, :, :, filter_index])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, model.input)[0]

        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([model.input], [loss, grads])

        # step size for gradient ascent
        step = 5.

        #img_asc = np.array(img)
        img_asc = np.random.random((1, 48, 48, 1))
        img_asc = (img_asc - 0.5) * 20 + 48
        
        # run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([img_asc])
            img_asc += grads_value * step

        img_asc = img_asc[0]
        img_ascs.append(deprocess_image(img_asc).reshape((48, 48)))
        
    if layer_output.shape[3] >= 35:
        plot_x, plot_y = 6, 6
    elif layer_output.shape[3] >= 23:
        plot_x, plot_y = 4, 6
    elif layer_output.shape[3] >= 11:
        plot_x, plot_y = 2, 6
    else:
        plot_x, plot_y = 1, 2
    
    fig, ax = plt.subplots(plot_x, plot_y, figsize = (12, 12))
    ax[0, 0].imshow(img.reshape((48, 48)), cmap = 'gray')
    ax[0, 0].set_title('Input image')
    fig.suptitle('Input image and %s filters' % (layer_name,))
    fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
    for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
        if x == 0 and y == 0:
            continue
        ax[x, y].imshow(img_ascs[x * plot_y + y - 1], cmap = 'gray')
        ax[x, y].set_title('filter %d' % (x * plot_y + y - 1))

    #fig.savefig('./result/image{}.png'.format(1), dpi=100)
    return img_ascs

def main():
    
    X_train = np.load('./feature/X_train.npy')
    Y_train = np.load('./feature/y_train.npy')
    X_train = normal(X_train)

    lb = preprocessing.LabelBinarizer()
    lb.fit(Y_train)
    Y_train = lb.inverse_transform(Y_train)

    #print(Y_train.shape)
    K.set_learning_phase(1)

    n_classes = 7
    model_name = "model-75.hdf5"
    model_path = "model-75.hdf5"    

    emotion_classifier = load_model(model_path)
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    input_img = emotion_classifier.input
    #print(layer_dict)
    #print(Y_train[0].shape)

    fig, ax = plt.subplots(7, 5, figsize = (16, 16))
    fig.suptitle('vanilla gradient')
    for i in range(n_classes):
        img = np.array(X_train[i+8])
        
        #Y_train[i] = np.reshape(Y_train[i],(7,1))
        vanilla = GradientSaliency(emotion_classifier, Y_train[i])
        mask = vanilla.get_mask(img)
        filter_mask = (mask > 0.0).reshape((48, 48))
        smooth_mask = vanilla.get_smoothed_mask(img)
        filter_smoothed_mask = (smooth_mask > 0.0).reshape((48, 48))

        ax[i, 0].imshow(img.reshape((48, 48 )), cmap = 'gray')
        cax = ax[i, 1].imshow(mask.reshape((48, 48)), cmap = 'jet')
        fig.colorbar(cax, ax = ax[i, 1])
        ax[i, 2].imshow(mask.reshape((48, 48)) * filter_mask, cmap = 'gray')
        cax = ax[i, 3].imshow(mask.reshape((48, 48)), cmap = 'jet')
        fig.colorbar(cax, ax = ax[i, 3])
        ax[i, 4].imshow(smooth_mask.reshape((48, 48)) * filter_smoothed_mask, cmap = 'gray')

    fig.savefig('image_Heatmap{}.png'.format(8), dpi=100)


if __name__ == "__main__":
    main()