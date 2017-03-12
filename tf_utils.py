import tensorflow as tf

def weight_var(shape, name=None):
    return tf.Variable(tf.random_normal(shape, stddev=0.02), name=name)

def bias_var(shape):
    return tf.Variable(tf.zeros(shape, dtype=tf.float32))