"""
Contrastive Divergence Learner for xRBM Models
"""
import tensorflow as tf

class CDApproximator():
    def __init__(self, learning_rate, momentum=0, k=1, 
                regularizer=None):
        self._cd_k = k
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._regularizer = regularizer
        self._momentum_vector = []


    def compute_gradients(self, cost, params, var_list=None): 
        grad_params = tf.gradients(ys=cost, xs=params)
        return grad_params

    def apply_gradients(self, model, grads):
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

        def _step(i, chain_sample):
            i = tf.add(i,1)
            _, chain_sample, _, _ = model.gibbs_sample_vhv(chain_sample, in_data)
            return i, chain_sample


        if len(self._momentum_vector) == 0:
            for param in model.model_params:
                self._momentum_vector.append(tf.Variable(param.initialized_value()))
        
        with tf.variable_scope('gibbs_sampling'):
            counter = tf.constant(0)
            c = lambda i, *args: tf.less(i, self._cd_k)
            [_,chain_end] = tf.while_loop(c, _step, [counter, vis_data], name='cd_loop')
         

        #return chain_end
        chain_end = tf.stop_gradient(chain_end)

        cost = model.get_cost(vis_data, chain_end, in_data)
        
        if self._regularizer is not None:
            with tf.name_scope('regularization'):
                weight_vars = [v for v in model.model_params if len(v.get_shape()) > 1]
                for W in weight_vars:
                    cost = cost + self._regularizer(W)
    

        grads = self.compute_gradients(cost, model.model_params, var_list=var_list)

        update_ops = self.apply_gradients(model, grads)

        return update_ops
    
        
