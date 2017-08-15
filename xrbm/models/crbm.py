"""
Conditional Restricted Boltzmann Machines (CRBM) Implementation in Tensorflow
"""

import tensorflow as tf
import numpy as np
import scipy.io as sio
from xrbm.utils import tfutils

class CRBM():
    'Conditional Restricted Boltzmann Machines (CRBM)'

    def __init__(self, num_vis, num_cond, num_hid, vis_type='binary',
                 activation=tf.nn.sigmoid,  
                 initializer=tf.contrib.layers.variance_scaling_initializer(), # He Init
                 name='CRBM'):
        """
        The CRBM Constructor

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
        self.num_cond = num_cond

        # Weights
        self.W = None
        self.A = None
        self.B = None

        # Biases
        self.vbias = None
        self.hbias = None

        # Learning params
        self.model_params = None

        with tf.variable_scope(self.name):
            self.create_variables()

    def create_variables(self):
        """
        Creates the variables used by the model
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
            
            self.vbias = tf.get_variable(shape=[self.num_vis], 
                                         initializer=tf.constant_initializer(0), 
                                         name='vbias')
            self.hbias = tf.get_variable(shape=[self.num_hid], 
                                         initializer=tf.constant_initializer(0), 
                                         name='hbias')

            self.model_params = [self.W, self.A, self.B, self.vbias, self.hbias]

    def sample_h_from_vc(self, visible, cond): 
        """
        Gets a sample of the hidden units, given tensors of visible and condition units

        Parameters
        ----------
        visible:    tensor
            a tensor of visible units configurations
        
        cond:    tensor
            a tensor of condition units configurations
        
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
            bottom_up = (tf.matmul(visible, self.W) + # visible to hidden 
                         tf.matmul(cond, self.B) +  # condition to hidden
                         self.hbias) # static hidden biases

            h_probs_means = self.activation(bottom_up)
            h_samples = tfutils.sample_bernoulli(h_probs_means)

        return bottom_up, h_probs_means, h_samples

    def sample_v_from_hc(self, hidden, cond):
        """
        Gets a sample of the visible units, given  tensors of hidden and condition units

        Parameters
        ----------
        hidden:    tensor
            a tensor of hidden units configurations

        cond:    tensor
            a tensor of condition units configurations
        
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
            contributions = (tf.matmul(hidden, tf.transpose(self.W), name='hidden_to_visible') + # hidden to visible
                          tf.matmul(cond, self.A, name='condition_to_visible') + # condition to visible
                          self.vbias) # static visible biases

            v_probs_means = self.activation(contributions)

            if self.vis_type == 'binary':                
                v_samples = tfutils.sample_bernoulli(v_probs_means)
            elif self.vis_type == 'gaussian':
                v_samples = contributions # using means instead of sampling, as in Taylor et al


        return contributions, v_probs_means, v_samples

    def gibbs_sample_hvh(self, h_samples0, cond):
        """
        Runs a cycle of gibbs sampling, started with an initial hidden units activations

        Used for training

        Parameters
        ----------
        h_samples0:    tensor
            a tensor of initial hidden units activations
        
        cond:    tensor
            a tensor of condition units configurations

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
            top_bottom, v_probs_means, v_samples = self.sample_v_from_hc(h_samples0, cond)

            # h from v
            bottom_up, h_probs_means, h_samples = self.sample_h_from_vc(v_samples, cond)

        return v_probs_means, v_samples, h_probs_means, h_samples

    def gibbs_sample_hvh_condcont(self, h_samples0, condontA, condontB):
        """
        Runs a cycle of gibbs sampling, started with an initial hidden units activations

        Uses pre-computed contributions from condition units to both hidden and visible units

        Parameters
        ----------
        h_samples0:    tensor
            a tensor of initial hidden units activations
        
        condontA:    tensor
            a tensor of contributions from condition to visible units

        condontB:    tensor
            a tensor of contributions from condition to hidden units

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
            contributions_to_vis = (tf.matmul(h_samples0, tf.transpose(self.W)) + # hidden to visible
                          condontA + # condition to visible
                          self.vbias) # static visible biases

            v_probs_means = self.activation(contributions_to_vis)

            if self.vis_type == 'binary':                
                v_samples = tfutils.sample_bernoulli(v_probs_means)
            elif self.vis_type == 'gaussian':
                v_samples = contributions_to_vis # using means instead of sampling, as in Taylor et al


            # h from v            
            contributions_to_hid = (tf.matmul(v_samples, self.W) + # visible to hidden 
                         condontB +  # condition to hidden
                         self.hbias) # static hidden biases

            h_probs_means = self.activation(contributions_to_hid)
            h_samples = tfutils.sample_bernoulli(h_probs_means)


        return v_probs_means, v_samples, h_probs_means, h_samples
    
    def gibbs_sample_vhv(self, v_samples0, in_data):
        """
        Runs a cycle of gibbs sampling, started with an initial visible and condition units

        Parameters
        ----------
        v_samples0:    tensor
            a tensor of initial visible units configuration
        
        cond:    tensor
            a tensor of condition units configurations

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
        cond = in_data[0]
        with tf.variable_scope('sampling_vhv'):
            # h from v
            bottom_up, h_probs_means, h_samples = self.sample_h_from_vc(v_samples0, cond)

            # v from h
            top_bottom, v_probs_means, v_samples = self.sample_v_from_hc(h_samples, cond)
        
        return v_probs_means, v_samples, h_probs_means, h_samples

    def get_cost(self, v_sample, chain_end, in_data):
        """
        Calculates the free-energy cost between two data tensors, used for calcuating the gradients
    
        Parameters
        ----------
        v_sample:   tensor
            the tensor A
        cond:   tensor
            the condition tensor
        chain_end:  tensor
            the tensor B

        Returns
        -------
        cost:       float
            the cost
        """
        cond = in_data[0]

        with tf.variable_scope('fe_cost'):
            cost = tf.reduce_mean(self.free_energy(v_sample, cond)
                    - self.free_energy(chain_end, cond), reduction_indices=0)
        return cost


    def free_energy(self, v_sample, cond):
        """
        Calcuates the free-energy of a given visible tensor

        Parameters
        ----------
        v_sample:   tensor
            the visible units tensor
        cond:       tensor
            the condition units tensor

        Returns
        -------
        e:  float
            the free energy
        """
        with tf.variable_scope('free_energy'):
            bottom_up = (tf.matmul(v_sample, self.W) + # visible to hidden 
                         tf.matmul(cond, self.B) + # condition to hidden
                         self.hbias) # static hidden biases
            
            vbias_n_cond = self.vbias + tf.matmul(cond, self.A)

            if self.vis_type == 'binary':
                v = - tf.matmul(v_sample, tf.expand_dims(vbias_n_cond,1), name='bin_visible_term')
            elif self.vis_type == 'gaussian':
                v = tf.reduce_sum(0.5 * tf.square(v_sample - vbias_n_cond), reduction_indices=1, name='gauss_visible_term')            

            h = - tf.reduce_sum(tf.log(1 + tf.exp(bottom_up)), reduction_indices=1, name='hidden_term')

        return tf.transpose(tf.transpose(v) + tf.transpose(h))        
    
    def predict(self, cond, init,  num_gibbs=5):
        """
        Generate (predict) the visible units configuration given the conditions

        Parameters
        ----------        
        cond:       tensor
            the condition units tensor
        init:       tensor
            the configuation to initialize the visible units with
        num_gibbs:  int, default 5
            the number of gibbs sampling cycles

        Returns
        -------
        sample:     tensor
            the predicted visible units
        hsample:    tensor
            the hidden units activations
        """

        # gibbs
        for k in range(num_gibbs): #TODO: this has to be more efficient since cond_data does not change
            vmean, sample, hmean, hsample = self.gibbs_sample_vhv(init, [cond])
            init = sample
        
        # mean-field approximation as suggested by Taylor
        _, vmean, sample = self.sample_v_from_hc(hmean, cond)

        return sample, hsample


