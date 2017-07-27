"""
Contrastive Divergence Learner for xRBM Models
"""
import tensorflow as tf

class CDLearner():
    def __init__(self, learning_rate, momentum=0, k=1):
        self.cd_k = k
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.momentum_vector = []


    def compute_gradients(self, cost, params, var_list=None):
        
        grad_params = tf.gradients(ys=cost, xs=params)

        return grad_params

    def apply_gradients(self, model, grads):
        update_ops = []
        mom_ops = []
        with tf.name_scope('CDLearning/updates'):
            for param, grad, mv in zip(model.model_params, grads, self.momentum_vector):
                mv = tf.assign(mv, self.momentum * mv + grad * self.learning_rate)
                update_ops.append(tf.assign_sub(param, mv))
                

             # compose the update values, incorporating weight decay, momentum, and sparsity terms
            #wu_ = tf.assign(self.wu, momentum * self.wu + (grad_params[0]  * self.W) * learning_rateW)
            #au_ = tf.assign(self.au, momentum * self.au + (grad_params[1] - wdecay * self.A) * learning_rateA)
            #bu_ = tf.assign(self.bu, momentum * self.bu + (grad_params[2] - wdecay * self.B) * learning_rateB)
            #vbu_ = tf.assign(self.vbu, momentum * self.vbu + grad_params[3] * learning_rate)
            #hbu_ = tf.assign(self.hbu, momentum * self.hbu + grad_params[4] * learning_rate)

            #momentum_ops = [wu_, au_, bu_, vbu_, hbu_]

             # ops to update the parameters
            #update_ops = [tf.assign_sub(self.W, self.wu),
            #              tf.assign_sub(self.A, self.au),
            #              tf.assign_sub(self.B, self.bu),
            #              tf.assign_sub(self.vbias, self.vbu),
            #              tf.assign_sub(self.hbias, self.hbu)]


        return update_ops, mom_ops

    def train(self, model, data, global_step=None, var_list=None, name=None):

        def _step(i, chain_sample):
            i = tf.add(i,1)
            chain_sample,_, _, _ = model.gibbs_sample_vhv(chain_sample, data[1:])
            return i, chain_sample


        if len(self.momentum_vector) == 0:
            for param in model.model_params:
                self.momentum_vector.append(tf.Variable(param.initialized_value()))
        
        with tf.variable_scope('gibbs_sampling'):
            counter = tf.constant(0)
            c = lambda i, *args: tf.less(i, self.cd_k)
            [_,chain_end] = tf.while_loop(c, _step, [counter, data[0]], name='cd_loop')
         

        #return chain_end
        chain_end = tf.stop_gradient(chain_end)

        cost = model.get_cost(data[0], chain_end)
        grads = self.compute_gradients(cost, model.model_params, var_list=var_list)

        update_ops = self.apply_gradients(model, grads)

        return update_ops
    
        
