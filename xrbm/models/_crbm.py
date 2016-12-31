# Conditional Restricted Boltzmann Machines (RBM) Implementation in Tensorflow
# Created by Omid Alemi

import tensorflow as tf
import numpy as np
from xrbm.utils import tfutils
from xrbm.utils import costs as costs
from .abstract_model import AbstractRBM

class CRBM(AbstractRBM):
    'Conditional Restricted Boltzmann Machines (RBM)'

    def __init__(self, num_vis, num_hid, orderA, orderB, vis_type='binary',
                 activation=tf.nn.sigmoid,  name='CRBM'):

        super(CRBM, self).__init__(num_vis, num_hid, vis_type, activation, name)

        # Model Params
        self.orderA = orderA
        self.orderB = orderB

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

    def create_placeholders_variables(self):
        # with tf.variable_scope(self.name):
        with tf.variable_scope('params'):
            self.W = tfutils.weight_variable([self.num_vis, self.num_hid], 'main_weights')
            self.A = tfutils.weight_variable([self.orderA * self.num_vis, self.num_vis], 'c2v_weights')
            self.B = tfutils.weight_variable([self.orderB * self.num_vis, self.num_hid], 'c2h_weights')
            self.vbias = tfutils.bias_variable([self.num_vis], 'vbias')
            self.hbias = tfutils.bias_variable([self.num_hid], 'hbias')

            self.model_params = [self.W, self.A, self.B, self.vbias, self.hbias]

        with tf.variable_scope('updates'):
            self.wu = tfutils.weight_variable([self.num_vis, self.num_hid], 'main_weights')
            self.au = tfutils.weight_variable([self.orderA * self.num_vis, self.num_vis], 'c2v_weights')
            self.bu = tfutils.weight_variable([self.orderB * self.num_vis, self.num_hid], 'c2h_weights')

            self.vbu = tfutils.bias_variable([self.num_vis], 'vbias')
            self.hbu = tfutils.bias_variable([self.num_hid], 'hbias')            

    def sample_h_from_v(self, visible, cond, hrand):
        with tf.variable_scope('sampling_hv'):        
            bottom_up = tf.matmul(visible, self.W) + # visible to hidden 
                        tf.matmul(cond, self.B) # condition to hidden
                        self.hbias # static hidden biases

            h_probs_means = self.activation(bottom_up)

            h_samples = tf.nn.relu(tf.sign(h_probs_means - hrand))

        return bottom_up, h_probs_means, h_samples

    def sample_h_from_v_np(self, v, c):
        with tf.variable_scope('sampling_hv'):
            vtf = tf.placeholder(tf.float32, shape=v.shape)
            cond = tf.placeholder(tf.float32, shape=c.shape)
            hrand = tf.placeholder(tf.float32, shape=(None, self.num_hid))

            _hrand = np.random.rand(v.shape[0], self.num_hid)

            a, h_probs_means, h_samples = self.sample_h_from_v(vtf, cond, hrand)

            init = tf.initialize_all_variables()
            with tf.Session() as sess:
                sess.run(init)
                m, s = sess.run([h_probs_means, h_samples], feed_dict={vtf:v, cond: v, hrand:_hrand})

        return m, s

    def sample_v_from_h(self, hidden, cond, vrand):
        with tf.variable_scope('sampling_vh'):
            top_bottom = tf.matmul(hidden, tf.transpose(self.W)) + # hidden to visible
                         tf.matmul(cond, self.A) # condition to visible
                         self.vbias # static visible biases

            v_probs_means = self.activation(top_bottom)

            if self.vis_type == 'binary':
                # v_samples = tf.to_float(tf.multinomial(v_probs_means, self.num_vis))
                v_samples = tf.nn.relu(tf.sign(v_probs_means - vrand))
            elif self.vis_type == 'gaussian':
                v_samples = top_bottom # using means instead of sampling, as in Taylor et al
            else:
                v_samples =  None

        return top_bottom, v_probs_means, v_samples

    def gibbs_sample_hvh(self, h_samples0, cond):
        with tf.variable_scope('sampling_hvh'):
            # v from h
            top_bottom, v_probs_means, v_samples = self.sample_v_from_h(h_samples0, cond, self.vrand)

            # h from v
            bottom_up, h_probs_means, h_samples = self.sample_h_from_v(v_samples, cond, self.hrand)

        return v_probs_means, v_samples, h_probs_means, h_samples

    def inference(self, input_data, cond, cd_k=1, sparse_target=0, sparse_cost=0, sparse_decay=0):
        'Define the tensorflow operations'

        with tf.variable_scope('inference'):
            ### positive phase
            # bottom-up - initialze the network with training data

            _, h_probs_means1, h_samples1 = self.sample_h_from_v(input_data, cond, self.hrand)

            chain_data = h_samples1

            # calculate the sparsity term
            current_activations_mean_props = tf.reduce_mean(h_probs_means1, reduction_indices=0)

            self.sp_hidden_means = sparse_decay * self.sp_hidden_means + (1 - sparse_decay) * current_activations_mean_props
            sparse_grad = sparse_cost * (self.sp_hidden_means - sparse_target)

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

            for k in range(cd_k):
                v_probs_means, v_samples, h_probs_means, h_samples = self.gibbs_sample_hvh(chain_data, cond)
                chain_data = h_samples

            ### update
            chain_end = v_samples

        return v_probs_means, chain_end, sparse_grad, current_activations_mean_props

    def train_step(self, visible_data, cond, learning_rate,
                   momentum=0, wdecay=0, cd_k=1,
                   sparse_target=0, sparse_cost=0, sparse_decay=0):
        
        learning_rateW = learning_rate
        learning_rateA = learning_rate
        learning_rateB = learning_rate
        
        with tf.variable_scope('train_step'):
            ## inference
            chain_end_probs_means, chain_end, sparse_grad, current_activations_mean_props = self.inference(visible_data, cond, 
                                                        cd_k, sparse_target, sparse_cost, sparse_decay)

            ## update
             # get the cost using free energy
            cost = self.get_cost(visible_data, chain_end)

             # calculate the gradients using tf
            grad_params = tf.gradients(ys=cost, xs=self.model_params)

             # compose the update values, incorporating weight decay, momentum, and sparsity terms
            wu_ = self.wu.assign(momentum * self.wu + (grad_params[0] + sparse_grad - wdecay * self.W) * learning_rateW)
            au_ = self.au.assign(momentum * self.au + (grad_params[1]  - wdecay * self.A) * learning_rateA)
            bu_ = self.bu.assign(momentum * self.bu + (grad_params[2]  - wdecay * self.B) * learning_rateB)
            vbu_ = self.vbu.assign(momentum * self.vbu + grad_params[3] * learning_rate)
            hbu_ = self.hbu.assign(momentum * self.hbu + (grad_params[4] + sparse_grad) * learning_rate)

            updates = [wu_, au_, bu_, vbu_, hbu_]

             # update the parameters
            w_ = self.W.assign_sub(self.wu)
            a_ = self.A.assign_sub(self.au)
            b_ = self.B.assign_sub(self.bu)
            vb_ = self.vbias.assign_sub(self.vbu)
            hb_ = self.hbias.assign_sub(self.hbu)

             # we need to return the new params so that tf considers them in the graph
            new_params = [w_, a_, b_, vb_, hb_]

            ## evaluate the reconstruction capability of the model
            rec_cost = self.get_reconstruction_cost(visible_data, chain_end_probs_means)

        return rec_cost, new_params, updates, current_activations_mean_props #TODO: Remove

    def get_cost(self, v_sample, chain_end):
        with tf.variable_scope('fe_cost'):
            cost = tf.reduce_mean(self.free_energy(v_sample)
                    - self.free_energy(chain_end), reduction_indices=0)
        return cost

    def get_reconstruction_cost(self, input_data, recon_means):
        cost = costs.cross_entropy(input_data, recon_means)
        return cost

    def get_reconstruction_cost_np(self, input_data, recon_data):
        i = tf.placeholder(tf.float32, shape=(None, self.num_vis))
        r = tf.placeholder(tf.float32, shape=(None, self.num_vis))
        c = self.get_reconstruction_cost(i, r)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            cost = sess.run(c, feed_dict={i:input_data, r: recon_data})
        return cost

    def free_energy(self, v_sample):        
        with tf.variable_scope('free_energy'):
            bottom_up = tf.matmul(v_sample, self.W) + self.hbias

            if self.vis_type == 'binary':
                v = - tf.matmul(v_sample, tf.expand_dims(self.vbias,1), name='bin_visible_term')
            elif self.vis_type == 'gaussian':
                v = tf.reduce_sum(0.5 * tf.square(v_sample - self.vbias), reduction_indices=1, name='gauss_visible_term')            

            h = - tf.reduce_sum(tf.log(1 + tf.exp(bottom_up)), reduction_indices=1, name='hidden_term')

        return tf.transpose(tf.transpose(v) + tf.transpose(h))        

    def free_energy_np(self, v_sample):
        v = tf.placeholder(tf.float32, shape=(None, self.num_vis))
        fe = self.free_energy(v)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            fe = sess.run(fe, feed_dict={v:v_sample})
        return fe

    def train(self, input_data, training_epochs, batch_size=100, learning_rate=0.1,
                    snapshot_dir='./logs/', snapshot_freq=100, cd_k=1,
                    momentum=0, wdecay=0, sparse_target=0, sparse_cost=0, sparse_decay=0):
        'Trains the RBM'

        # Create the tf tensors
        with tf.variable_scope(self.name):
            self.create_placeholders_variables()
            self.sp_hidden_means = tf.get_variable(name='sp_hidden_means',
                                                   shape=[self.num_hid],
                                                   initializer=tf.constant_initializer(sparse_target))

        # Make batches
        batch_idxs = np.random.permutation(range(len(input_data)))
        n_batches = len(batch_idxs) // batch_size
        with tf.name_scope(self.name+'_ops'):
            mom = tf.Variable(initial_value=momentum, dtype=tf.float32)
            sp_cost = tf.Variable(initial_value=momentum, dtype=tf.float32)
            batch_data_ph = tfutils.data_variable((None,input_data.shape[1]),'batch_data')
            train_step = self.train_step(batch_data_ph, learning_rate, mom, wdecay, cd_k=cd_k,
                            sparse_target=sparse_target, sparse_cost=sp_cost, sparse_decay=sparse_decay)

        saver = tf.train.Saver()

        # Run everything in tf
        with tf.Session() as sess:
            train_writer = tf.train.SummaryWriter(snapshot_dir, sess.graph)

            sess.run(tf.initialize_all_variables())

            for epoch in range(training_epochs):
                epoch_cost = 0
                epoch_h_means = 0;

                m = momentum
                sc = sparse_cost
                if epoch < 5:
                    m = 0
                # if epoch < 5:
                    # sc = 0

                for batch_i in range(n_batches):
                    # Get just minibatch amount of data
                    idxs_i = batch_idxs[batch_i * batch_size:(batch_i + 1) * batch_size]

                    (rec_cost, new_params, updates, h_means) = sess.run(train_step, feed_dict={batch_data_ph:input_data[idxs_i],
                                                            self.hrand: np.random.rand(input_data[idxs_i].shape[0], self.num_hid),
                                                            self.vrand: np.random.rand(input_data[idxs_i].shape[0], self.num_vis),
                                                            mom: m, sp_cost: sc})
                    epoch_cost += rec_cost
                    epoch_h_means += h_means

                print('Epoch %i / %i | cost = %f | momentum = %f | sparse cost = %f'%(epoch+1, training_epochs, epoch_cost, m, sc))

            save_path = saver.save(sess, '%s%s_model.ckpt' % (snapshot_dir, self.name))

            return self.W.eval(), self.vbias.eval(), self.hbias.eval()
