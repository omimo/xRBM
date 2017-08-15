"""
Restricted Boltzmann Machines (RBM) Implementation in Tensorflow
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
from xrbm.utils import tfutils

class RBM():
    'Restricted Boltzmann Machines (RBM)'

    def __init__(self, num_vis, num_hid, vis_type='binary',
                 activation=tf.nn.sigmoid,  
                 initializer=tf.contrib.layers.variance_scaling_initializer(), # He Init
                 name='RBM'):
        """
        The RBM Constructor

        Parameters
        ----------
        num_vis:       int
            number of visible input units
        num_hid:       int
            number of hidden units
        vis_type:      string
            the type of the visible units (`binary` or `gaussian`)
        activation:    callable f(h)
            a function reference to the activation function of the hidden units
        initializer:   callable f(h)
            a function reference to the weight initializer (TF-style)
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
        
        # Weights
        self.W = None

        # Biases
        self.vbias = None
        self.hbias = None

        # Learning params
        self.model_params = None

        with tf.variable_scope(self.name):
            self.create_variables()
    
    def create_variables(self):
        """
        Creates the TF variables used by the model
        """
        with tf.variable_scope('params'):
            self.W = tf.get_variable(shape=[self.num_vis, self.num_hid], 
                                     initializer=self.initializer,
                                     name='main_weights')
            
            self.vbias = tf.get_variable(shape=[self.num_vis], initializer=tf.constant_initializer(0), name='vbias')
            self.hbias = tf.get_variable(shape=[self.num_hid], initializer=tf.constant_initializer(0), name='hbias')

            self.model_params = [self.W, self.vbias, self.hbias]


    def sample_h_from_v(self, visible):
        """
        Gets a sample of hidden units, given a tensor of visible units configuations

        Parameters
        ----------
        visible:    tensor
            a tensor of visible units configurations
        
        Returns
        -------
        bottom_up:      tensor
            a tensor containing the bottom up contributions before activation
        
        h_prob_means:   tensor
            a tensor containing the mean probabilities of the hidden units activations

        h_samples:      tensor
            a tensor containing a bernoulli sample generated from the mean activations
        """     
        with tf.variable_scope('sampling_hv'):
            bottom_up = tf.matmul(visible, self.W) + self.hbias
            h_probs_means = self.activation(bottom_up)
            h_samples = tfutils.sample_bernoulli(h_probs_means)

        return bottom_up, h_probs_means, h_samples

    def sample_v_from_h(self, hidden):
        """
        Get a sample of visible units, given a tensor of hidden units configuations

        Parameters
        ----------
        hidden:    tensor
            a tensor of hidden units configurations
        
        Returns
        -------
        top_bottom:      tensor
            a tensor containing the top bottom contributions
        
        v_probs_means:   tensor
            a tensor containing the mean probabilities of the visible units

        v_samples:      tensor
            a tensor containing a sample of visible units generated from the top bottom contributions
        """     
        with tf.variable_scope('sampling_vh'):
            top_bottom = tf.matmul(hidden, tf.transpose(self.W)) + self.vbias # or hidden * W^T

            v_probs_means = self.activation(top_bottom)

            if self.vis_type == 'binary':                
                v_samples = tfutils.sample_bernoulli(v_probs_means)
            elif self.vis_type == 'gaussian':
                v_samples = top_bottom # using means instead of sampling, as in Taylor et al
            else:
                v_samples =  None

        return top_bottom, v_probs_means, v_samples

    def gibbs_sample_hvh(self, h_samples0):    
        """
        Runs a cycle of gibbs sampling, starting with an initial hidden units activations

        Parameters
        ----------
        h_samples0:    tensor
            a tensor of initial hidden units activations
        
        Returns
        -------
        v_probs_means:  tensor

        v_samples:      tensor
            visible samples
        h_probs_means:  tensor
            a tensor containing the mean probabilities of the hidden units activations
        h_samples:      tensor
            a tensor containing a bernoulli sample generated from the mean activations
        """
        with tf.variable_scope('sampling_hvh'):
            # v from h
            top_bottom, v_probs_means, v_samples = self.sample_v_from_h(h_samples0)

            # h from v
            bottom_up, h_probs_means, h_samples = self.sample_h_from_v(v_samples)

        return v_probs_means, v_samples, h_probs_means, h_samples

    def gibbs_sample_vhv(self, v_samples0, *data):    
        """
        Runs a cycle of gibbs sampling, starting with an initial visible data

        Parameters
        ----------
        v_samples0:    tensor
            a tensor of visible units values
        
        Returns
        -------
        v_probs_means:  tensor

        v_samples:      tensor
            visible samples
        h_probs_means:  tensor
            a tensor containing the mean probabilities of the hidden units activations
        h_samples:      tensor
            a tensor containing a bernoulli sample generated from the mean activations
        """
        with tf.variable_scope('sampling_vhv'):
            # h from v
            bottom_up, h_probs_means, h_samples = self.sample_h_from_v(v_samples0)

            # v from h
            top_bottom, v_probs_means, v_samples = self.sample_v_from_h(h_samples)

        return v_probs_means, v_samples, h_probs_means, h_samples


    def get_cost(self, v_sample, chain_end, in_data=[]):
        """
        Calculates the free-energy cost between two data tensors, used for calcuating the gradients
    
        Parameters
        ----------
        v_sample:   tensor
            the tensor A
        chain_end:  tensor
            the tensor B

        Returns
        -------
        cost:       float
            the cost
        """
        with tf.variable_scope('fe_cost'):
            cost = tf.reduce_mean(self.free_energy(v_sample)
                    - self.free_energy(chain_end), reduction_indices=0)
        return cost

    def free_energy(self, v_sample): 
        """
        Calcuates the free-energy of a given visible tensor

        Parameters
        ----------
        v_sample:   tensor
            the visible units tensor

        Returns
        -------
        e:  float
            the free energy
        """

        with tf.variable_scope('free_energy'):
            bottom_up = tf.matmul(v_sample, self.W) + self.hbias

            if self.vis_type == 'binary':
                v = - tf.matmul(v_sample, tf.expand_dims(self.vbias,1), name='bin_visible_term')
            elif self.vis_type == 'gaussian':
                v = tf.reduce_sum(0.5 * tf.square(v_sample - self.vbias), reduction_indices=1, name='gauss_visible_term')            

            h = - tf.reduce_sum(tf.log(1 + tf.exp(bottom_up)), reduction_indices=1, name='hidden_term')

        return tf.transpose(tf.transpose(v) + tf.transpose(h))        
