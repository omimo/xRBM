"""
An example of how to train an RBM with more control over training parameters

Created by Omid Alemi
"""
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from xrbm.models.rbm import RBM
from xrbm.utils import tfutils
import xrbm.train


data_sets = input_data.read_data_sets('MNIST_data', False)

# Set up the parameters
training_data = data_sets.train.images
snapshot_dir = './logs/'
snapshot_freq = 100
num_vis = training_data[0].shape[0]
num_hid = 200
learning_rate=0.05
batch_size=100
cd_k=1
momentum=0.9
training_epochs=30
activation=tf.nn.elu
initializer = tf.contrib.layers.xavier_initializer()
regularizer = None#tf.contrib.layers.l1_regularizer(0.001)

r1 = RBM(num_vis=num_vis, num_hid=num_hid, vis_type='binary', 
                  name='rbm_mnist_custom', activation=activation, initializer=initializer)


mom = tf.Variable(initial_value=momentum, dtype=tf.float32)
batch_data_ph = tfutils.data_variable((None,training_data.shape[1]),'batch_data')

reccost_op = r1.get_reconstruction_cost(batch_data_ph)

cdapproximator = xrbm.train.CDApproximator(0.05, momentum=mom, k=10, regularizer=regularizer)

train_op = cdapproximator.train(r1, vis_data=batch_data_ph)

# Make batches
r1.batch_size = batch_size
batch_idxs = np.random.permutation(range(len(training_data)))
n_batches = len(batch_idxs) // batch_size

with tf.Session() as sess:   
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        epoch_cost = 0
        if epoch < 6:
            m = 0
        else:
            m = momentum

        for batch_i in range(n_batches):
            idxs_i = batch_idxs[batch_i * batch_size:(batch_i + 1) * batch_size]
            # Create the feed for the batch data
            feed = feed_dict={batch_data_ph: training_data[idxs_i], mom:m}

            # Run the training step
            sess.run(train_op, feed_dict=feed)

        rec_cost = sess.run(reccost_op, feed_dict={batch_data_ph: training_data})
        epoch_cost = rec_cost

        print('Epoch %i / %i | cost = %f | lr = %f | momentum = %f'%(epoch+1, training_epochs, epoch_cost, learning_rate, m)) 
