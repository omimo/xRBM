"""
Contrastive Divergence Gradient Approximator
"""

import tensorflow as tf

class CDApproximator():
    """
    Contrastive Divergence Gradient Approximator
    """
    def __init__(self, learning_rate, momentum=0, k=1, 
                regularizer=None):
        self._cd_k = k
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._regularizer = regularizer
        self._momentum_vector = []


    def compute_gradients(self, cost, params, var_list=None):
        """
        Computes the gradients of the given cost function w.r.t the tensors in the params
        """

        grad_params = tf.gradients(ys=cost, xs=params)
        return grad_params

    def apply_updates(self, model, grads):
        """
        Updates the model parameters based on the given gradients, using momentum
        """
        update_ops = []
        mom_ops = []
        
        if isinstance(self._learning_rate, list):
            lrs = self._learning_rate
            print('d')
        else:
            lrs = [self._learning_rate for p in model.model_params]

        with tf.name_scope('CDLearning/updates'):
            for param, grad, mv, lr in zip(model.model_params, grads, self._momentum_vector, lrs):
                mv = tf.assign(mv, self._momentum * mv + grad * lr)
                update_ops.append(tf.assign_sub(param, mv))
                
        return update_ops, mom_ops

    def train(self, model, vis_data, in_data=[], global_step=None, var_list=None, name=None):
        """
        Performs one step of the CD-k algorithm to approximate the model parameters
        """
        

        # Internal function to perform one step of gibbs sampling and increase the loop counter
        def _step(i, chain_sample):
            i = tf.add(i,1)
            _, chain_sample, _, _ = model.gibbs_sample_vhv(chain_sample, in_data)
            return i, chain_sample

        # If first time, fill the momentum vector
        if len(self._momentum_vector) == 0:
            for param in model.model_params:
                self._momentum_vector.append(tf.Variable(param.initialized_value()))
        
        # Perform k steps of gibbs sampling and store the last sample in chain_end
        with tf.variable_scope('gibbs_sampling'):
            counter = tf.constant(0) # loop counter
            c = lambda i, *args: tf.less(i, self._cd_k) # loop condition
            [_,chain_end] = tf.while_loop(c, _step, [counter, vis_data], name='cd_loop')
         

        # Since we don't want TF to calculate the gradients for the whole chain, we stop it!
        chain_end = tf.stop_gradient(chain_end)
        
        # Get the model's cost function for the training data and the reconstructed data (chain_end)
        cost = model.get_cost(vis_data, chain_end, in_data)
        
        # We a regularizer is set, then add the regularization terms to the cost function
        if self._regularizer is not None:
            with tf.name_scope('regularization'):
                # We only apply the regularization on weights
                # We can assume that weight tensors are 2D and biases are 1D
                weight_vars = [v for v in model.model_params if len(v.get_shape()) > 1]
                for W in weight_vars:
                    cost = cost + self._regularizer(W)
    
        # Compute the gradients
        grads = self.compute_gradients(cost, model.model_params, var_list=var_list)
        
        # Update the model parameters
        update_ops = self.apply_updates(model, grads)

        return update_ops
    
        
