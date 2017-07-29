'''
Cost functions for xRBM Library
Created by Omid Alemi - 2016
'''

import tensorflow as tf
import numpy as np

def cross_entropy(dataA, dataB):                    
        a = dataA * tf.log(tf.sigmoid(dataB))
        b = (1 - dataA) * tf.log(1 - tf.sigmoid(dataB))
        cross_entropy = tf.reduce_mean(tf.reduce_sum(a+b, reduction_indices=1), reduction_indices=0)
        return cross_entropy

def mse(dataA, dataB):
        loss = tf.reduce_mean(tf.square(dataA - dataB))
        return loss