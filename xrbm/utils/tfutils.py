import tensorflow as tf
import numpy as np


def sample_bernoulli(means):    
    #if n==-1:
    #    n = tf.shape(means)[0]
    #shape = [n , means.get_shape().as_list()[1]] #[n, means.get_shape()[1]]
    shape = tf.shape(means)
    return tf.where(means - tf.random_uniform(shape) > 0, 
                                  tf.ones(shape), 
                                  tf.zeros(shape))

def _sample_bernoulli(means, n=-1):
    tf.nn.relu(tf.sign(means - tf.random_uniform(tf.shape(means))))

# def sample_bernoulli(means, n):
#     rand = np.random.rand(n, means.get_shape().as_list()[1])    
#     return tf.nn.relu(tf.sign(means - rand)) 
    
