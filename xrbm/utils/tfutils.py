import tensorflow as tf
import numpy as np

def weight_variable(shape, name='weight'):
    initial = tf.truncated_normal(shape, stddev=0.1) #TODO: right choice?
    # return tf.Variable(initial, name=name)
    return tf.get_variable(name=name, shape=shape,
                        #    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1) 
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=4 * np.sqrt(6. / (shape[0] + shape[1])))
                        #    initializer = initial
                           )

def bias_variable(shape, name='bias', initializer=tf.constant_initializer(0.0)):  
    # initial = tf.constant(0.1, shape=shape)
    # return tf.Variable(initial, name=name)
    return tf.get_variable(name=name, shape=shape, initializer=initializer)

def data_variable(shape, name='input_data'):
    return tf.placeholder(tf.float32, shape=shape, name=name)


def sample_bernoulli(means, n=-1):    
    if n==-1:
        n = means.get_shape().as_list()[0]
    shape = [n , means.get_shape().as_list()[1]] #[n, means.get_shape()[1]]
    return tf.select(means - tf.random_uniform(shape) > 0, 
                                  tf.ones(shape), 
                                  tf.zeros(shape))

# def sample_bernoulli(means, n):
#     rand = np.random.rand(n, means.get_shape().as_list()[1])    
#     return tf.nn.relu(tf.sign(means - rand)) 
    