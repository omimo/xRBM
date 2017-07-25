"""
CRestricted Boltzmann Machines (CRBM) Implementation in Tensorflow
"""

import tensorflow as tf
import numpy as np
import scipy.io as sio
from xrbm.utils import tfutils
from xrbm.utils import costs as costs
from .abstract_model import AbstractRBM

class CRBM(AbstractRBM):
    'Conditional Restricted Boltzmann Machines (CRBM)'

    def __init__(self, num_vis, num_cond, num_hid, vis_type='binary',
                 activation=tf.nn.sigmoid,  
                 initializer=tf.contrib.layers.variance_scaling_initializer(), # He Init
                 W_regularizer=None,
                 A_regularizer=None,
                 B_regularizer=None,
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
        name:          string
            the name of the object (used for Tensorflow's scope)
        """
        super(CRBM, self).__init__(num_vis, num_hid, vis_type, activation, initializer, name)

        # Model Params
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
        self.wu = None
        self.au = None
        self.bu = None
        self.vbu = None
        self.hbu = None

        # Regularizer
        self.W_regularizer = W_regularizer
        self.A_regularizer = A_regularizer
        self.B_regularizer = B_regularizer

        with tf.variable_scope(self.name):
            self.create_placeholders_variables()

    def create_placeholders_variables(self):
        """
        Creates 
        """
        with tf.variable_scope(self.name):
            self.sp_hidden_means = tf.get_variable(name='sp_hidden_means',
                                                   shape=[self.num_hid],
                                                   initializer=tf.constant_initializer(0.2))

            self.input_data = tfutils.data_variable((None, self.num_vis),'input_data')
            self.cond_data = tfutils.data_variable((None, self.num_cond),'condition_data')

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

        with tf.variable_scope('updates'):
            self.wu = tfutils.weight_variable([self.num_vis, self.num_hid], 'main_weights')
            self.au = tfutils.weight_variable([self.num_cond, self.num_vis], 'c2v_weights')
            self.bu = tfutils.weight_variable([self.num_cond, self.num_hid], 'c2h_weights')

            self.vbu = tfutils.bias_variable([self.num_vis], 'vbias', initializer=tf.constant_initializer(0.0))
            self.hbu = tfutils.bias_variable([self.num_hid], 'hbias')           


    def sample_h_from_vc(self, visible, cond, n=-1): 
        """
        Get a sample of hidden units, given a tensor of visible and condition units configuations

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
            h_samples = tfutils.sample_bernoulli(h_probs_means, n)

        return bottom_up, h_probs_means, h_samples

    def sample_v_from_hc(self, hidden, cond, n=-1):
        """
        Get a sample of visible units, given a tensor of hidden and condition units configuations

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
            contributions = (tf.matmul(hidden, tf.transpose(self.W)) + # hidden to visible
                          tf.matmul(cond, self.A) + # condition to visible
                          self.vbias) # static visible biases

            v_probs_means = self.activation(contributions)

            if self.vis_type == 'binary':                
                v_samples = tfutils.sample_bernoulli(v_probs_means, n)
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

    def gibbs_sample_hvh_condcont(self, h_samples0, condontA, condontB, n):
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
                v_samples = tfutils.sample_bernoulli(v_probs_means, n)
            elif self.vis_type == 'gaussian':
                v_samples = contributions_to_vis # using means instead of sampling, as in Taylor et al


            # h from v            
            contributions_to_hid = (tf.matmul(v_samples, self.W) + # visible to hidden 
                         condontB +  # condition to hidden
                         self.hbias) # static hidden biases

            h_probs_means = self.activation(contributions_to_hid)
            h_samples = tfutils.sample_bernoulli(h_probs_means, n)


        return v_probs_means, v_samples, h_probs_means, h_samples
    
    def gibbs_sample_vhv(self, v_samples0, cond, n):
        """
        Runs a cycle of gibbs sampling, started with an initial visible and condition units configurations

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
        with tf.variable_scope('sampling_vhv'):
            # h from v
            bottom_up, h_probs_means, h_samples = self.sample_h_from_vc(v_samples0, cond)

            # v from h
            top_bottom, v_probs_means, v_samples = self.sample_v_from_hc(h_samples, cond)

        return v_probs_means, v_samples, h_probs_means, h_samples

    def inference(self, input_data, cond_data, cd_k=1):
        """
        Defines the tensorflow operations for inference

        Parameters
        ----------
        input_data:     tensor
            the input (batch) data tensor
        cond_data:      tensor
            the condition data tensor
        cd_k=1:         int, default 1
            the number of CD steps for gibbs sampling

        Returns
        -------
        v_probs_means:  tensor
            a tensor containing the mean probabilities of the visible units
        chain_end:      tensor
            the last visible samples generated in the gibbs cycles
        """

        with tf.variable_scope('inference'):
            ### positive phase
            # bottom-up - initialze the network with training data
            
            _, h_probs_means1, h_samples1 = self.sample_h_from_vc(input_data, cond=cond_data)

            chain_data = h_samples1


            # v_samples = tf.stop_gradient(v_samples)

#            condcontA = tf.matmul(cond_data, self.A)
#            condcontB = tf.matmul(cond_data, self.B)
            for k in range(cd_k): #TODO: this has to be more efficient since cond_data does not change
#                v_probs_means, v_samples, h_probs_means, h_samples = self.gibbs_sample_hvh_condcont(chain_data, 
#                                                            condcontA,
#                                                            condcontB,
#                                                            self.batch_size)

                v_probs_means, v_samples, h_probs_means, h_samples = self.gibbs_sample_hvh(chain_data, 
                                                            cond_data)

                chain_data = h_samples

            ### update
            chain_end = v_samples

        return v_probs_means, chain_end

    def train_step(self, visible_data, cond_data, learning_rate,
                   momentum=0, wdecay=0, cd_k=1):
        """
        Defines the operations needed for a training step of a CRBM

        Parameters
        ----------
        visible_data:       tensor
            the input (batch) data tensor
        cond_data:      tensor
            the condition data tensor
        learning_rate:      float
            the learning rate
        momentum:           float, default 0
            the momentum value
        wdecay:             float, default 0
            the weight decay value
        cd_k:               int, default 1
            the number of CD steps for gibbs sampling

        Returns
        -------
        rec_cost:       float
            the reconstruction cost for this step
        new_params:     list of tensors
            the updated model parameters
        updates:        list of tensors
            the value of updates for each model parameter
        """
        learning_rateW = learning_rate
        learning_rateA = learning_rate * 0.01 # it's a hack, the autoregressive weights often need a smaller lr
        learning_rateB = learning_rate

        with tf.variable_scope('train_step'):
            ## inference
            chain_end_probs_means, chain_end = self.inference(visible_data, cond_data, cd_k)

            ## update
             # get the cost using free energy
            cost = self.get_cost(visible_data, cond_data, chain_end)

            # regularize
            if self.W_regularizer is not None:
                cost = cost + self.W_regularizer(self.W)

            if self.A_regularizer is not None:
                cost = cost + self.A_regularizer(self.A)

            if self.B_regularizer is not None:
                cost = cost + self.B_regularizer(self.B)

             # calculate the gradients using tf
            grad_params = tf.gradients(ys=cost, xs=self.model_params)

             # compose the update values, incorporating weight decay, momentum, and sparsity terms
            wu_ = tf.assign(self.wu, momentum * self.wu + (grad_params[0] - wdecay * self.W) * learning_rateW)
            au_ = tf.assign(self.au, momentum * self.au + (grad_params[1] - wdecay * self.A) * learning_rateA)
            bu_ = tf.assign(self.bu, momentum * self.bu + (grad_params[2] - wdecay * self.B) * learning_rateB)
            vbu_ = tf.assign(self.vbu, momentum * self.vbu + grad_params[3] * learning_rate)
            hbu_ = tf.assign(self.hbu, momentum * self.hbu + grad_params[4] * learning_rate)

            momentum_ops = [wu_, au_, bu_, vbu_, hbu_]

             # ops to update the parameters
            update_ops = [tf.assign_sub(self.W, self.wu),
                          tf.assign_sub(self.A, self.au),
                          tf.assign_sub(self.B, self.bu),
                          tf.assign_sub(self.vbias, self.vbu),
                          tf.assign_sub(self.hbias, self.hbu)]


        return [momentum_ops, update_ops]

    def get_cost(self, v_sample, cond, chain_end):
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

        with tf.variable_scope('fe_cost'):
            cost = tf.reduce_mean(self.free_energy(v_sample, cond)
                    - self.free_energy(chain_end, cond), reduction_indices=0)
        return cost

    def get_reconstruction_cost(self, input_data, cond_data):
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
        recon_means,_,_,_ = self.gibbs_sample_vhv(input_data, cond_data, tf.shape(input_data)[0])

        #cost = costs.cross_entropy(input_data, recon_means)
        cost = costs.mse(input_data, recon_means)
        return cost

    def free_energy(self, v_sample, cond):  #TODO: change     
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
    
    def make_prediction(self, cond, init,  num_gibbs=20):
        """
        Generate (predict) the visible units configuration given the conditions

        Parameters
        ----------        
        cond:       tensor
            the condition units tensor
        init:       tensor
            the configuation to initialize the visible units with
        num_gibbs:  int, default 20
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
            vmean, sample, hmean, hsample = self.gibbs_sample_vhv(init, cond, 1)
            init = sample
        
        # mean-field approximation as suggested by Taylor
        _, vmean, sample = self.sample_v_from_hc(hmean, cond, 1)

        return sample, hsample


    def train(self, sess, input_data, cond_data, training_epochs, batch_size=100, learning_rate=0.1,
                    snapshot_dir='./logs/', snapshot_freq=0, cd_k=1,
                    momentum=0, wdecay=0):
        """
        Creates mini-batches and trains the CRBM for the given number of epochs

        Parameters
        ----------
        input_data:         tensor
            the input data tensor
        cond_data:         tensor
            the condition data tensor
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

        Returns
        -------
        W:      array_like
            the numpy visble-hidden weight matrix
        A:      array_like
            the numpy condition to visible weight matrix
        B:      array_like
            the numpy condition to hidden weight matrix
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
        train_op = self.train_step(self.input_data,
                                   self.cond_data, 
                                   learning_rate, 
                                   momentum, 
                                   wdecay, 
                                   cd_k=cd_k)
        
        reccost_op = self.get_reconstruction_cost(self.input_data, self.cond_data)

        saver = tf.train.Saver()

        # Run everything in tf 
        for epoch in range(training_epochs):
            epoch_cost = 0

            for batch_i in range(n_batches):
                # Get just minibatch amount of data
                idxs_i = batch_idxs[batch_i * batch_size:(batch_i + 1) * batch_size]
                
                # Add noise to the past, Gaussian with std 1
                cd_noise = cond_data[idxs_i] + np.random.normal(0, 1, [batch_size, self.num_cond])

                # Create the feed for the batch data
                feed = feed_dict={self.input_data: input_data[idxs_i],
                                  self.cond_data: cd_noise}

                # Run the training step
                sess.run(train_op, feed_dict=feed)

            
            epoch_cost = sess.run(reccost_op, feed_dict={self.input_data: input_data[idxs_i], self.cond_data: cond_data[idxs_i]})

            print('Epoch %i / %i | cost = %f | lr = %f | momentum = %f'%
                 (epoch+1, training_epochs, epoch_cost, learning_rate, momentum))
                    
            if snapshot_freq != 0 and (epoch+1) % snapshot_freq == 0:                
                save_path = saver.save(sess, '%s%s_ep%i_model.ckpt' % (snapshot_dir, self.name, (epoch+1)))

        return self.W.eval(session=sess), self.A.eval(session=sess), self.B.eval(session=sess), self.vbias.eval(session=sess), self.hbias.eval(session=sess)
