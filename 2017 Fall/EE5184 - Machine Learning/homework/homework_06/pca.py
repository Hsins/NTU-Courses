import numpy as np
import skimage
from skimage import io
import os,sys 
import glob
#import matplotlib.pyplot as plt

def read_images_in_folder(path):
    image_stack = []
    for img in glob.glob(path+'/*.jpg'): # All jpeg images
        image_stack.append(io.imread(img))
    image_stack = np.asarray(image_stack)
    return image_stack

def average_img(images):
    #from PIL import Image
    image_mean = np.array(np.mean(images, axis=(0)))
    #out = Image.fromarray(image_mean.astype(np.uint8),'RGB')
    #out.show()
    #out.save('average_img.jpg')
    return image_mean

def reconstruction(U_tr):
    U_tr -= np.min(U_tr)
    U_tr /= np.max(U_tr)
    U_tr *= 255
    U_tr = U_tr.astype(np.uint8)
    return U_tr
'''
def eigenface():
    U_reconstruct = reconstruction(U_tr[3])
    re_U_reconstruct = np.reshape(U_reconstruct,(600, 600, 3))
    io.imsave('eigenface4.jpg',re_U_reconstruct)
    imgplot=plt.imshow(re_U_reconstruct)
    plt.show()
'''
#def main():
def main(*args):
    #path ='Aberdeen'
    path = args[0][1]
    image_stack = read_images_in_folder(path)
    H = image_stack.shape[1]
    W = image_stack.shape[2]

    image_flatten=[i.flatten() for i in image_stack]
    #print(len(image_flatten))
    #print(len(image_flatten[0]))

    image_mean = average_img(image_stack)
    image_mean_flatten=image_mean.flatten()

    image_norm = np.zeros([len(image_flatten), len(image_flatten[0])])
    image_norm = (image_flatten - image_mean_flatten)

    image_norm_tr=np.transpose(image_norm)
    U, s, V=np.linalg.svd(image_norm_tr, full_matrices=False)

    U_tr=np.array(np.transpose(U))

    eigen = 4
    #input_image = '1.jpg'
    input_image = args[0][2]

    img_reconstruct = io.imread(path+'/'+input_image)
    img_reconstruct = np.asarray(img_reconstruct)
    img_reconstruct_flatten = img_reconstruct.flatten()
    img_reconstruct_flatten = img_reconstruct_flatten - image_mean_flatten

    weight = np.inner(img_reconstruct_flatten,U_tr[:eigen])
    img_reconstruct = np.inner(U_tr[:eigen].T,weight)
    img_reconstruct += image_mean_flatten

    img_reconstruct = reconstruction(img_reconstruct)

    img_reconstruct.shape
    re_img_reconstruct = np.reshape(img_reconstruct,(H, W, 3))
    io.imsave('reconstruction.jpg',re_img_reconstruct)

if __name__ == '__main__':
    main(sys.argv)
    #main()

#imgplot=plt.imshow(re_img_reconstruct)
#plt.show()