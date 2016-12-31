# Restricted Boltzmann Machines (RBM) Implementation in Tensorflow
# Created by Omid Alemi

import tensorflow as tf
import numpy as np
import scipy.io as sio
from xrbm.utils import tfutils
from xrbm.utils import costs as costs
from .abstract_model import AbstractRBM

class CRBM(AbstractRBM):
    'Conditional Restricted Boltzmann Machines (CRBM)'

    def __init__(self, num_vis, num_cond, num_hid, vis_type='binary',
                 activation=tf.nn.sigmoid,  name='CRBM'):
        super(CRBM, self).__init__(num_vis, num_hid, vis_type, activation, name)

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

        with tf.variable_scope(self.name):
            self.create_placeholders_variables()

    def create_placeholders_variables(self):
        with tf.variable_scope(self.name):
            self.sp_hidden_means = tf.get_variable(name='sp_hidden_means',
                                                   shape=[self.num_hid],
                                                   initializer=tf.constant_initializer(0.2))

            self.input_data = tfutils.data_variable((None, self.num_vis),'input_data')
            self.cond_data = tfutils.data_variable((None, self.num_cond),'condition_data')

        with tf.variable_scope('params'):
            self.W = tfutils.weight_variable([self.num_vis, self.num_hid], 'main_weights')
            self.A = tfutils.weight_variable([self.num_cond, self.num_vis], 'c2v_weights')
            self.B = tfutils.weight_variable([self.num_cond, self.num_hid], 'c2h_weights')
            self.vbias = tfutils.bias_variable([self.num_vis], 'vbias')
            self.hbias = tfutils.bias_variable([self.num_hid], 'hbias')

            self.model_params = [self.W, self.A, self.B, self.vbias, self.hbias]

        with tf.variable_scope('updates'):
            self.wu = tfutils.weight_variable([self.num_vis, self.num_hid], 'main_weights')
            self.au = tfutils.weight_variable([self.num_cond, self.num_vis], 'c2v_weights')
            self.bu = tfutils.weight_variable([self.num_cond, self.num_hid], 'c2h_weights')

            self.vbu = tfutils.bias_variable([self.num_vis], 'vbias', initializer=tf.constant_initializer(0.0))
            self.hbu = tfutils.bias_variable([self.num_hid], 'hbias', initializer=tf.constant_initializer(-0.1))           


    def sample_h_from_vc(self, visible, cond, n=-1):        
        with tf.variable_scope('sampling_hv'):
            bottom_up = (tf.matmul(visible, self.W) + # visible to hidden 
                         tf.matmul(cond, self.B) +  # condition to hidden
                         self.hbias) # static hidden biases

            h_probs_means = self.activation(bottom_up)
            h_samples = tfutils.sample_bernoulli(h_probs_means, n)

        return bottom_up, h_probs_means, h_samples

    def sample_v_from_hc(self, hidden, cond, n=-1):
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

    def gibbs_sample_hvh(self, h_samples0, cond, n):        
        with tf.variable_scope('sampling_hvh'):
            # v from h
            top_bottom, v_probs_means, v_samples = self.sample_v_from_hc(h_samples0, cond, n)

            # h from v
            bottom_up, h_probs_means, h_samples = self.sample_h_from_vc(v_samples, cond, n)

        return v_probs_means, v_samples, h_probs_means, h_samples

    def gibbs_sample_hvh_condcont(self, h_samples0, condontA, condontB, n):        
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
        with tf.variable_scope('sampling_vhv'):
            # h from v
            bottom_up, h_probs_means, h_samples = self.sample_h_from_vc(v_samples0, cond, n)

            # v from h
            top_bottom, v_probs_means, v_samples = self.sample_v_from_hc(h_samples, cond, n)

        return v_probs_means, v_samples, h_probs_means, h_samples

    def inference(self, input_data, cond_data, cd_k=1, sparse_target=0, sparse_cost=0, sparse_decay=0):
        'Define the tensorflow operations'

        with tf.variable_scope('inference'):
            ### positive phase
            # bottom-up - initialze the network with training data
            
            _, h_probs_means1, h_samples1 = self.sample_h_from_vc(input_data, cond=cond_data, n=self.batch_size)

            chain_data = h_samples1

            # calculate the sparsity term
            # current_activations_mean_props = tf.reduce_mean(h_probs_means1, reduction_indices=0)

            # self.sp_hidden_means = sparse_decay * current_activations_mean_props + (1 - sparse_decay) * self.sp_hidden_means
            # sparse_grad = sparse_cost * (sparse_target - self.sp_hidden_means)
            
            current_activations_mean_props = h_probs_means1
            sparse_grad = 0

            ### negative phase
            # def gibbs_step(k, chain_data):
            #     tf.add(k, 1)
            #     return self.gibbs_sample_hvh(chain_data)
                

            # # perform k steps of gibbs
            # k = tf.constant(0)
            # cond = lambda k: tf.less(k, cd_k)
            # # b = lambda k: tf.add(k, 1)
            # # r = tf.while_loop(cond, b, [k])

            # v_probs_means, v_samples, h_probs_means, h_samples = tf.while_loop(cond, gibbs_step, [k, chain_data])

            # for k in range(cd_k): #TODO: this has to be more efficient since cond_data does not change
            #     v_probs_means, v_samples, h_probs_means, h_samples = self.gibbs_sample_hvh(chain_data, cond_data, self.batch_size)
            #     chain_data = h_samples
            
            # v_samples = tf.stop_gradient(v_samples)

            condcontA = tf.matmul(cond_data, self.A)
            condcontB = tf.matmul(cond_data, self.B)
            for k in range(cd_k): #TODO: this has to be more efficient since cond_data does not change
                v_probs_means, v_samples, h_probs_means, h_samples = self.gibbs_sample_hvh_condcont(chain_data, 
                                                            condcontA,
                                                            condcontB,
                                                            self.batch_size)
                chain_data = h_samples

            ### update
            chain_end = v_samples

        return v_probs_means, chain_end, sparse_grad, current_activations_mean_props

    def train_step(self, visible_data, cond_data, learning_rate,
                   momentum=0, wdecay=0, cd_k=1,
                   sparse_target=0, sparse_cost=0, sparse_decay=0):
        
        learning_rateW = learning_rate
        learning_rateA = learning_rate * 0.01
        learning_rateB = learning_rate

        with tf.variable_scope('train_step'):
            ## inference
            (chain_end_probs_means, 
            chain_end, 
            sparse_grad, 
            current_activations_mean_props) = self.inference(visible_data, cond_data, 
                                                             cd_k, sparse_target, sparse_cost, sparse_decay)

            ## update
             # get the cost using free energy
            cost = self.get_cost(visible_data, cond_data, chain_end)

             # calculate the gradients using tf
            grad_params = tf.gradients(ys=cost, xs=self.model_params)

             # compose the update values, incorporating weight decay, momentum, and sparsity terms
            wu_ = self.wu.assign(momentum * self.wu - (grad_params[0] - sparse_grad - wdecay * self.W) * learning_rateW)
            au_ = self.au.assign(momentum * self.au - (grad_params[1] - wdecay * self.A) * learning_rateA)
            bu_ = self.bu.assign(momentum * self.bu - (grad_params[2] - wdecay * self.B) * learning_rateB)
            vbu_ = self.vbu.assign(momentum * self.vbu - grad_params[3] * learning_rate)
            hbu_ = self.hbu.assign(momentum * self.hbu - (grad_params[4] - sparse_grad) * learning_rate)

            updates = [wu_, au_, bu_, vbu_, hbu_]

             # update the parameters
            w_ = self.W.assign_add(self.wu)
            a_ = self.A.assign_add(self.au)
            b_ = self.B.assign_add(self.bu)
            vb_ = self.vbias.assign_add(self.vbu)
            hb_ = self.hbias.assign_add(self.hbu)

             # we need to return the new params so that tf considers them in the graph
            new_params = [w_, a_, b_, vb_, hb_]

            ## evaluate the reconstruction capability of the model
            rec_cost = self.get_reconstruction_cost(visible_data, chain_end_probs_means)

        return rec_cost, new_params, updates, current_activations_mean_props #TODO: Remove

    def get_cost(self, v_sample, cond, chain_end):
        with tf.variable_scope('fe_cost'):
            cost = tf.reduce_mean(self.free_energy(v_sample, cond)
                    - self.free_energy(chain_end, cond), reduction_indices=0)
        return cost

    def get_reconstruction_cost(self, input_data, recon_means):
        cost = costs.cross_entropy(input_data, recon_means)
        return cost

    def free_energy(self, v_sample, cond):  #TODO: change     
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
        'Generate (predict) the network input given the conditions'

        # gibbs
        for k in range(num_gibbs): #TODO: this has to be more efficient since cond_data does not change
            vmean, sample, hmean, hsample = self.gibbs_sample_vhv(init, cond, 1)
            init = sample
        
        # mean-field approximation as suggested by Taylor
        _, vmean, sample = self.sample_v_from_hc(hmean, cond, 1)

        return sample, hsample


    def train(self, sess, input_data, cond_data, training_epochs, batch_size=100, learning_rate=0.1,
                    snapshot_dir='./logs/', snapshot_freq=0, cd_k=1,
                    momentum=0, wdecay=0, sparse_target=0, sparse_cost=0, sparse_decay=0):
        'Trains the RBM'

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
                                   cd_k=cd_k,
                                   sparse_target=sparse_target, 
                                   sparse_cost=sparse_cost, 
                                   sparse_decay=sparse_decay)
        
        saver = tf.train.Saver()

        # Run everything in tf 
        for epoch in range(training_epochs):
            epoch_cost = 0
            epoch_h_means = 0;

            for batch_i in range(n_batches):
                # Get just minibatch amount of data
                idxs_i = batch_idxs[batch_i * batch_size:(batch_i + 1) * batch_size]
                
                # Add noise to the past, Gaussian with std 1
                cd_noise = cond_data[idxs_i] + np.random.normal(0, 1, [batch_size, self.num_cond])

                # Create the feed for the batch data
                feed = feed_dict={self.input_data: input_data[idxs_i],
                                  self.cond_data: cd_noise}

                # Run the training step
                (rec_cost, new_params, updates, h_means) = sess.run(train_op, feed_dict=feed)

                # Add up the cost
                epoch_cost += rec_cost
                epoch_h_means += h_means
            
            epoch_cost = epoch_cost/n_batches
            print('Epoch %i / %i | cost = %f | lr = %f | momentum = %f | sparse cost = %f'%
                 (epoch+1, training_epochs, epoch_cost, learning_rate, momentum, sparse_cost))
                    
            if snapshot_freq != 0 and (epoch+1) % snapshot_freq == 0:
                print('Saving snapshot')
                sio.savemat('snap_ep%i.mat'%(epoch+1), {'W': self.W.eval(session=sess), 
                                                        'A': self.A.eval(session=sess),
                                                        'B': self.B.eval(session=sess), 
                                                        'vb': self.vbias.eval(session=sess), 
                                                        'hb': self.hbias.eval(session=sess)})     
        
                save_path = saver.save(sess, '%s%s_ep%i_model.ckpt' % (snapshot_dir, self.name, (epoch+1)))

        return self.W.eval(session=sess), self.A.eval(session=sess), self.B.eval(session=sess), self.vbias.eval(session=sess), self.hbias.eval(session=sess)
