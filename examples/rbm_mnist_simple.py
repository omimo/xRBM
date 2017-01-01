"""
A simple example of how to train an RBM on image data

Created by Omid Alemi
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import PIL.Image as Image

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from xrbm.models.rbm import RBM
from xrbm.utils.vizutils import tile_raster_images


data_sets = input_data.read_data_sets('MNIST_data', False)

training_data = data_sets.train.images

 # Set up the parameters
snapshot_dir = './logs/'
snapshot_freq = 100
num_vis = training_data[0].shape[0]
num_hid = 300
learning_rate=0.1
batch_size=100
cd_k=5
wdecay=0.0001
momentum=0
training_epochs=15
activation=tf.nn.sigmoid
input_data = data_sets.train.images

print('-'*80)
print('Training RBM with %i %s units'%(num_hid, activation))
print('lr: %1.3f, batchsize: %i, cd: %i, wdecay: %f, mom: %1.1f'%(learning_rate, batch_size, cd_k, wdecay, momentum))
print('-'*80)


r1 = RBM(num_vis=num_vis, num_hid=num_hid, vis_type='binary', name='rbm_mnist_simple', activation=activation)


with tf.Session() as sess: 
    init = tf.global_variables_initializer()
    sess.run(init)

    W, vb, hb, = r1.train(sess, 
            input_data=input_data,
            training_epochs=training_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            cd_k=cd_k,
            wdecay=wdecay,
            momentum=momentum)
    
 
    image = Image.fromarray(
                tile_raster_images(
                    X=W.transpose(),
                    img_shape=(28, 28),
                    tile_shape=(15, 20),
                    tile_spacing=(1, 1)
                )
            )

    image.save('%s%s_filters_at_end.png' % (snapshot_dir, r1.name))
