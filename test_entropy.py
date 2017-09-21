import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python import debug as tf_debug
import vect_entropy_funcs as vf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    
    u_vals = tf.random_normal([100, int(3e4)], mean=0, stddev=1) #problems when n = 100
    u_val = u_vals[:,0]
    
    num_bins=50
    value_range = [tf.reduce_min(u_val), tf.reduce_max(u_val)]
    hist = tf.histogram_fixed_width(u_val, value_range=value_range, nbins=num_bins)
    bin_edges = tf.linspace(start=value_range[0], stop=value_range[1], num=num_bins)
    hist_final = tf.to_float(tf.divide(hist, tf.reduce_sum(hist)))
    
    unit_entropies, entropy_m0, entropy = sess.run(vf.unit_entropy(bin_edges, hist_final))
