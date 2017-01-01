"""
An example of how to train an RBM with more control over training parameters

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
from xrbm.utils import tfutils

data_sets = input_data.read_data_sets('MNIST_data', False)

 # Set up the parameters
training_data = data_sets.train.images
snapshot_dir = './logs/'
snapshot_freq = 100
num_vis = training_data[0].shape[0]
num_hid = 100
learning_rate=0.1
batch_size=100
cd_k=10
wdecay=0.0002
momentum=0.9
training_epochs=100
activation=tf.nn.relu
sparse_target=0.2
sparse_decay=0.99
sparse_cost=0


print('-'*80)
print('Training RBM with %i %s units'%(num_hid, activation))
print('lr: %1.3f, batchsize: %i, cd: %i, wdecay: %f, mom: %1.1f'%(learning_rate, batch_size, cd_k, wdecay, momentum))
print('-'*80)


r1 = RBM(num_vis=num_vis, num_hid=num_hid, vis_type='binary', name='rbm_mnist_custom', activation=activation)

# Create the tf tensors
r1.create_placeholders_variables()

# Make batches
r1.batch_size = batch_size
batch_idxs = np.random.permutation(range(len(training_data)))
n_batches = len(batch_idxs) // batch_size

# Define train ops            

mom = tf.Variable(initial_value=momentum, dtype=tf.float32)
sp_cost = tf.Variable(initial_value=momentum, dtype=tf.float32)
batch_data_ph = tfutils.data_variable((None,training_data.shape[1]),'batch_data')
lr = tf.Variable(initial_value=learning_rate, dtype=tf.float32)

train_op = r1.train_step(batch_data_ph, 
                            lr, 
                            mom, 
                            wdecay, 
                            cd_k=cd_k,
                            sparse_target=sparse_target, 
                            sparse_cost=sp_cost, 
                            sparse_decay=sparse_decay)

# Run everything in tf
with tf.Session() as sess:    

    sess.run(tf.global_variables_initializer())

    # Run everything in tf 
    for epoch in range(training_epochs):
        epoch_cost = 0
        epoch_h_means = 0;

        m = momentum
        sc = sparse_cost

        if epoch < 6:
            m = 0
            sc = 0
        
        # if epoch > 20:
        #     learning_rate = 0.001

        for batch_i in range(n_batches):
            # Get just minibatch amount of data
            idxs_i = batch_idxs[batch_i * batch_size:(batch_i + 1) * batch_size]

            # Create the feed for the batch data
            feed = feed_dict={batch_data_ph: training_data[idxs_i],
                              mom: m,
                              sp_cost: sc,
                              lr: learning_rate}

            # Run the training step
            (rec_cost, new_params, updates, h_means) = sess.run(train_op, feed_dict=feed)

            # Add up the cost
            epoch_cost += rec_cost
            epoch_h_means += h_means
        
        epoch_cost = epoch_cost/n_batches
        print('Epoch %i / %i | cost = %f | lr = %f | momentum = %f | sparse cost = %f'%
                (epoch+1, training_epochs, epoch_cost, learning_rate, m, sc))

        _W = r1.W.eval().transpose()

        image = Image.fromarray(
            tile_raster_images(
                X=_W,
                img_shape=(28,28),
                tile_shape=(15, 20),
                tile_spacing=(1, 1)
            )
        )
        image.save('%s%s_filters_at_epoch_%i.png' % (snapshot_dir, r1.name, epoch+1))