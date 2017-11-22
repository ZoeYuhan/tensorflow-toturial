#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Zoe
Created on Wed Aug  9 10:16:30 2017

"""

import cv2
import os
import vgg
import numpy as np
import h5py
import tensorflow as tf


#TODO: load VGG and extract feature
root="../continue/days_4_resize/"
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
vgg_weights, vgg_mean_pixel = vgg.load_net(VGG_PATH)
#CONTENT_LAYERS = ('relu3_1','relu3_2', 'relu4_1', 'relu5_1','relu5_2')
layer='relu5_2'
#input_image=cv2.imread("../intervel/days_2_resize/2_300001_2016-02-19_2016-02-22.png")
feature_data=[]
input_image_shape=(120, 160, 3)
shape = (1,) + input_image_shape
g = tf.Graph()
i=0
with g.as_default(),  tf.Session() as sess:
   
    image = tf.placeholder('float', shape=shape)
    net = vgg.net_preloaded(vgg_weights, image, 'avg')
    for name in os.listdir(root):
        i=i+1
        filename=os.path.join(root,name)
        input_image=cv2.imread(filename)
    #    content_features = {}
#    for i in range(len(image_list)):
#        input_image=image_list[i]
        content_pre = np.array([vgg.preprocess(input_image, vgg_mean_pixel)])
        content_features = net[layer].eval(feed_dict={image: content_pre})[0]
        feature_data.append(content_features.flatten())
        if i%100==0:
            print ("%s images feature extract" %i)
        
        
h5f=h5py.File(root+'feature.h5','w')
h5f.create_dataset('feature',data=feature_data)
h5f.close()

