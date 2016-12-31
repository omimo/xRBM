# Abstract Class for the Implementation of the Family of
# Restricted Boltzmann Machines (RBM) in Tensorflow
# Created by Omid Alemi

import tensorflow as tf
import numpy as np
from xrbm.utils import tfutils

class AbstractRBM(object):
    'Abstract Class for xRBM Models'

    def __init__(self, num_vis, num_hid, vis_type='binary',
                 activation=tf.nn.sigmoid,  name='xRBM'):

        # Model Params
        self.num_vis = num_vis
        self.num_hid = num_hid
        self.vis_type = vis_type
        self.name = name
        self.activation = activation

        # Training        
        self.sp_hidden_means = None

        # Data
        self.input_data = None
        self.data_mean = None
        self.data_std = None
        self.gsd = None

        # Weights

        # Biases

        # Learning params
        self.model_params = None

        # tf stuff
        self.sess = None

    def restore_model(self, sess, checkpoint_file):

        saver = tf.train.Saver()

        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint_file)
        print("Model restored.")