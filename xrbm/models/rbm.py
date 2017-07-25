"""
Restricted Boltzmann Machines (RBM) Implementation in Tensorflow
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
from xrbm.utils import tfutils
from xrbm.utils import costs as costs
from .abstract_model import AbstractRBM

class RBM(AbstractRBM):
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
        name:          string
            the name of the object (used for Tensorflow's scope)
        """
        super(RBM, self).__init__(num_vis, num_hid, vis_type, 
                                  activation=activation, 
                                  initializer=initializer, 
                                  name=name)

        # Weights
        self.W = None

        # Biases
        self.vbias = None
        self.hbias = None

        # Learning params
        self.model_params = None
        self.wu = None
        self.vbu = None
        self.hbu = None

        with tf.variable_scope(self.name):
            self.create_placeholders_variables()
    
    def create_placeholders_variables(self):
        """
        Creates the TF placeholders and variables used by the object
        """
        with tf.variable_scope(self.name):
            self.sp_hidden_means = tf.get_variable(name='sp_hidden_means',
                                                   shape=[self.num_hid],
                                                   initializer=tf.constant_initializer(0))

            self.batch_data = tfutils.data_variable((None, self.num_vis),'batch_data')

        with tf.variable_scope('params'):
            self.W = tf.get_variable(shape=[self.num_vis, self.num_hid], 
                                     initializer=self.initializer,
                                     name='main_weights')
            #self.W = tfutils.weight_variable([self.num_vis, self.num_hid], 'main_weights')
            self.vbias = tfutils.bias_variable([self.num_vis], 'vbias')
            self.hbias = tfutils.bias_variable([self.num_hid], 'hbias')

            self.model_params = [self.W, self.vbias, self.hbias]

        with tf.variable_scope('updates'):
            self.wu = tfutils.weight_variable([self.num_vis, self.num_hid], 'main_weights')
            self.vbu = tfutils.bias_variable([self.num_vis], 'vbias')
            self.hbu = tfutils.bias_variable([self.num_hid], 'hbias')           

    def load_model_params(self, _W, _vbias, _hbias):
        """
        Loads the model parameters from Numpy arrays

        Parameters
        ----------
        _W:     array_like
            The weight matrix
        _vbias: array_like
            The visible biases
        _hbias: array_like
            The hidden biases 
        """
        with tf.variable_scope(self.name):
            with tf.variable_scope('params'):
                self.W = tfutils.weight_variable([self.num_vis, self.num_hid], 'main_weights') #TODO: add reuse parameter
                self.vbias = tfutils.bias_variable([self.num_vis], 'vbias')
                self.hbias = tfutils.bias_variable([self.num_hid], 'hbias')

                self.model_params = [self.W, self.vbias, self.hbias]
                self.W.assign(_W)
                self.vbias.assign(_vbias)
                self.hbias.assign(_hbias)

    def sample_h_from_v(self, visible, n=-1):
        """
        Get a sample of hidden units, given a tensor of visible units configuations

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
            h_samples = tfutils.sample_bernoulli(h_probs_means, n)

        return bottom_up, h_probs_means, h_samples

    def sample_v_from_h(self, hidden, n=-1):
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
                v_samples = tfutils.sample_bernoulli(v_probs_means, n)
            elif self.vis_type == 'gaussian':
                v_samples = top_bottom # using means instead of sampling, as in Taylor et al
            else:
                v_samples =  None

        return top_bottom, v_probs_means, v_samples

    def gibbs_sample_hvh(self, h_samples0):    
        """
        Runs a cycle of gibbs sampling, started with an initial hidden units activations

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
            top_bottom, v_probs_means, v_samples = self.sample_v_from_h(h_samples0, n=self.batch_size)

            # h from v
            bottom_up, h_probs_means, h_samples = self.sample_h_from_v(v_samples, n=self.batch_size)

        return v_probs_means, v_samples, h_probs_means, h_samples

    def gibbs_sample_vhv(self, v_samples0):    
        """
        Runs a cycle of gibbs sampling, started with an initial hidden units activations

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
            bottom_up, h_probs_means, h_samples = self.sample_h_from_v(v_samples0, n=self.batch_size)

            # v from h
            top_bottom, v_probs_means, v_samples = self.sample_v_from_h(h_samples, n=self.batch_size)



        return v_probs_means, v_samples, h_probs_means, h_samples

    def inference(self, input_data, cd_k=1, sparse_target=0, sparse_cost=0, sparse_decay=0):
        """
        Defines the tensorflow operations for inference

        Parameters
        ----------
        input_data:     tensor
            the input (batch) data tensor
        cd_k=1:         int, default 1
            the number of CD steps for gibbs sampling
        sparse_target:  float, default 0
            the sparsity target
        sparse_cost:    float, default 0
            the sparsity cost
        sparse_decay:   float, default 0
            the sparsity weight decay

        Returns
        -------
        v_probs_means:  tensor
            a tensor containing the mean probabilities of the visible units
        chain_end:      tensor
            the last visible samples generated in the gibbs cycles
        sparse_grad:    tensor
            the sparisity gradients
        current_activations_mean_props:     tensor
            the mean activation of the hidden units
        """

        with tf.variable_scope('inference'):
            ### positive phase
            # bottom-up - initialze the network with training data
            
            _, h_probs_means1, h_samples1 = self.sample_h_from_v(input_data, n=self.batch_size)

            chain_data = h_samples1

            # calculate the sparsity term
            current_activations_mean_props = tf.reduce_mean(h_probs_means1, reduction_indices=0)

            self.sp_hidden_means = sparse_decay * self.sp_hidden_means + (1 - sparse_decay) * current_activations_mean_props
            sparse_grad = sparse_cost * (self.sp_hidden_means - sparse_target)

            for k in range(cd_k):
                v_probs_means, v_samples, h_probs_means, h_samples = self.gibbs_sample_hvh(chain_data)
                chain_data = h_samples

            ### update
            chain_end = v_samples

        return v_probs_means, chain_end, sparse_grad, current_activations_mean_props

    def train_step(self, visible_data, learning_rate,
                   momentum=0, wdecay=0, cd_k=1,
                   sparse_target=0, sparse_cost=0, sparse_decay=0):
        """
        Defines the operations needed for a training step of an RBM

        Parameters
        ----------
        visible_data:       tensor
            the input (batch) data tensor
        learning_rate:      float
            the learning rate
        momentum:           float, default 0
            the momentum value
        wdecay:             float, default 0
            the weight decay value
        cd_k:               int, default 1
            the number of CD steps for gibbs sampling
        sparse_target:  float, default 0
            the sparsity target
        sparse_cost:    float, default 0
            the sparsity cost
        sparse_decay:   float, default 0
            the sparsity weight decay

        Returns
        -------
        rec_cost:       float
            the reconstruction cost for this step
        new_params:     list of tensors
            the updated model parameters
        updates:        list of tensors
            the value of updates for each model parameter
        """
        with tf.variable_scope('train_step'):
            ## inference
            chain_end_probs_means, chain_end, sparse_grad, current_activations_mean_props = self.inference(visible_data, 
                                                            cd_k,
                                                            sparse_target, 
                                                            sparse_cost, 
                                                            sparse_decay)

            ## update
             # get the cost using free energy
            cost = self.get_cost(visible_data, chain_end)

             # calculate the gradients using tf
            grad_params = tf.gradients(ys=cost, xs=self.model_params)

             # compose the update values, incorporating weight decay, momentum, and sparsity terms
            wu_ = tf.assign(self.wu, momentum * self.wu + (grad_params[0] + sparse_grad - wdecay * self.W) * learning_rate)
            vbu_ = tf.assign(self.vbu, momentum * self.vbu + grad_params[1] * learning_rate)
            hbu_ = tf.assign(self.hbu, momentum * self.hbu + (grad_params[2] + sparse_grad) * learning_rate)

            momentum_ops = [wu_, 
                            vbu_, 
                            hbu_]

             # ops to update the parameters
            update_ops = [tf.assign_sub(self.W, self.wu), 
                          tf.assign_sub(self.vbias, self.vbu), 
                          tf.assign_sub(self.hbias, self.hbu)]

             # we need to return the new params so that tf considers them in the graph


            ## evaluate the reconstruction capability of the model
            #rec_cost = self.get_reconstruction_cost(visible_data, chain_end_probs_means)

        return [momentum_ops, update_ops]

    def get_cost(self, v_sample, chain_end):
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

    def get_reconstruction_cost(self, input_data):
        """
        Calculates the reconstruction cost between input data and reconstructed data
    
        Parameters
        ----------
        input_data:   tensor
            the input data tensor
        recon_means:  tensor
            the reconstructed data tensor

        Returns
        -------
        cost:       float
            the reconstruction cost
        """        
        recon_means,_,_,_ = self.gibbs_sample_vhv(input_data)

        # cost = costs.cross_entropy(input_data, recon_means)
        cost = costs.mse(input_data, recon_means)
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


    def train(self, sess, input_data, training_epochs, batch_size=100, learning_rate=0.1,
                    snapshot_dir='./logs/', snapshot_freq=100, cd_k=1,
                    momentum=0, wdecay=0, sparse_target=0, sparse_cost=0, sparse_decay=0):
        """
        Creates mini-batches and trains the RBM for the given number of epochs

        Parameters
        ----------
        input_data:         tensor
            the input data tensor
        training_epochs:    float
            the number of training epochs
        batch_size:         int
            the size of each mini batch
        learning_rate:      float, default 0.1
            the learning rate
        snapshot_dir:       string, default logs
            the directory to store model snapshots and logs
        snapshot_freq:      int, default 100
            the frequency of the epochs to save a model snapshot
        cd_k:               int, default 1
            the number of CD steps for gibbs sampling
        momentum:           float, default 0
            the momentum value
        wdecay:             float, default 0
            the weight decay value
        sparse_target:  float, default 0
            the sparsity target
        sparse_cost:    float, default 0
            the sparsity cost
        sparse_decay:   float, default 0
            the sparsity weight decay

        Returns
        -------
        W:      array_like
            the numpy weight matrix
        vbias:      array_like
            the numpy visible biases
        hbias:      array_like
            the numpy hidden biases
        """
        self.batch_size = batch_size

        # Make batches
        batch_idxs = np.random.permutation(range(len(input_data)))
        n_batches = len(batch_idxs) // batch_size
        
        # Define train ops            
        train_op = self.train_step(self.batch_data, 
                                   learning_rate, 
                                   momentum, 
                                   wdecay, 
                                   cd_k=cd_k,
                                   sparse_target=sparse_target, 
                                   sparse_cost=sparse_cost, 
                                   sparse_decay=sparse_decay)
 
        # Run everything in tf 
        for epoch in range(training_epochs):
            epoch_cost = 0
            epoch_h_means = 0;

            for batch_i in range(n_batches):
                # Get just minibatch amount of data
                idxs_i = batch_idxs[batch_i * batch_size:(batch_i + 1) * batch_size]

                # Create the feed for the batch data
                feed = feed_dict={self.batch_data: input_data[idxs_i]}

                # Run the training step
                sess.run(train_op, feed_dict=feed)
                
                # Get the cost
                

                # Add up the cost
                epoch_cost += rec_cost
                # epoch_h_means += h_means
            
            epoch_cost = epoch_cost/n_batches
            print('Epoch %i / %i | cost = %f | lr = %f | momentum = %f | sparse cost = %f'%
                 (epoch+1, training_epochs, epoch_cost, learning_rate, momentum, sparse_cost))
            
        # save_path = saver.save(sess, '%s%s_model.ckpt' % (snapshot_dir, self.name))

        return self.W.eval(session=sess), self.vbias.eval(session=sess), self.hbias.eval(session=sess)
