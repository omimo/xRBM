import tensorflow as tf
import numpy as np


def sample_bernoulli(means):    
    shape = tf.shape(means)
    return tf.where(means - tf.random_uniform(shape) > 0, 
                                  tf.ones(shape), 
                                  tf.zeros(shape))
