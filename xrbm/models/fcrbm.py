"""
Factored Conditional Restricted Boltzmann Machines (FCRBM) Implementation in Tensorflow
"""

import tensorflow as tf
import numpy as np
import scipy.io as sio
from xrbm.utils import tfutils
from xrbm.utils import costs as costs

class FCRBM():
    'Factored Conditional Restricted Boltzmann Machines (CRBM)'

    def __init__(self, num_vis, num_cond, num_hid, vis_type='binary',
                 activation=tf.nn.sigmoid,  
                 initializer=tf.contrib.layers.variance_scaling_initializer(), # He Init
                 name='CRBM'):
        """
        The FCRBM Constructor

        Parameters
        ----------
        num_vis:       int
            number of visible input units
        num_cond:       int
            number of condition units
        num_hid:       int
            number of hidden units
        vis_type:      string
            the type of the visible units (`binary` or `gaussian`)
        activation:    callable f(h)
            a function reference to the activation function of the hidden units
        name:          string
            the name of the object (used for Tensorflow's scope)
        """
        
        # Model Param
        self.num_vis = num_vis
        self.num_hid = num_hid
        self.vis_type = vis_type
        self.name = name
        self.activation = activation
        self.initializer = initializer
        self.num_cond = num_cond

        # Weights
        self.W_if = None
        self.W_of = None
        self.W_hf = None

        # Biases
        self.vbias = None
        self.hbias = None

        # Learning params
        self.model_params = None

        with tf.variable_scope(self.name):
            self.create_placeholders_variables()

    def create_placeholders_variables(self):
        """
        Creates 
        """
        with tf.variable_scope('params'):
            self.W = tf.get_variable(shape=[self.num_vis, self.num_hid], 
                                     initializer=self.initializer,
                                     name='vh_weights')


            self.A = tf.get_variable(shape=[self.num_cond, self.num_vis], 
                                     initializer=self.initializer,
                                     name='c2v_weights')

            self.B = tf.get_variable(shape=[self.num_cond, self.num_hid], 
                                     initializer=self.initializer,
                                     name='c2h_weights')
            #self.W = tfutils.weight_variable([self.num_vis, self.num_hid], 'main_weights')
            #self.A = tfutils.weight_variable([self.num_cond, self.num_vis], 'c2v_weights')
            #self.B = tfutils.weight_variable([self.num_cond, self.num_hid], 'c2h_weights')
            self.vbias = tfutils.bias_variable([self.num_vis], 'vbias')
            self.hbias = tfutils.bias_variable([self.num_hid], 'hbias')

            self.model_params = [self.W, self.A, self.B, self.vbias, self.hbias]

