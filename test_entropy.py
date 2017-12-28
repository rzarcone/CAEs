import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
import utils.entropy_funcs as ef

num_tests = 10
u_batch = 100
num_u = int(3e4)
u_mean = 0
u_stddev = 0.5
num_tri = 20
mle_lr = 0.1
num_mle = 5

graph = tf.Graph()
with graph.as_default():
    u_vals = tf.placeholder(tf.float32, shape=(u_batch, num_u), name="u_vals")
    tri_centers = tf.placeholder(tf.float32, shape=(num_tri), name="tri_centers")
    thetas, theta_init = ef.construct_thetas(num_u, num_tri)
    ll = ef.log_likelihood(u_vals, thetas, tri_centers)
    mle_update = ef.mle(ll, thetas, mle_lr)
    u_probs = ef.prob_est(u_vals, thetas, tri_centers)
    entropies = ef.calc_entropy(u_probs)
    ent_loss = tf.reduce_sum(entropies, name="entropy_loss")
    init_op = tf.global_variables_initializer()

mle_triangle_centers = np.linspace(-1.0, 1.0, num_tri)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config, graph=graph) as sess:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    
    for _ in range(num_tests):
      #gen_u_vals = np.random.normal(size=[u_batch, num_u], loc=u_mean, scale=u_stddev)
      gen_u_vals = np.random.laplace(size=[u_batch, num_u], loc=u_mean, scale=u_stddev)
      
      feed_dict = {u_vals:gen_u_vals, tri_centers:mle_triangle_centers}
      sess.run(init_op, feed_dict)
      for _ in range(num_mle):
        sess.run(mle_update, feed_dict)
      loss = sess.run(ent_loss, feed_dict)
      print(loss)
