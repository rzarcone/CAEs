import numpy as np
import tensorflow as tf
import time as ti

device = "/gpu:0"

def tf_get_line_eq(x_points, y_points):
  m = tf.divide(tf.subtract(y_points[1], y_points[0]), tf.subtract(x_points[1], x_points[0]))
  b = tf.subtract(y_points[1], tf.multiply(m, x_points[1]))
  return (m,b)

def np_get_line_eq(x_points, y_points):
  m = (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
  b = y_points[1] - m * x_points[1]
  return (m,b)

graph = tf.Graph()
with graph.as_default(), tf.device(device):
  tf_y_points = tf.placeholder(tf.float32)
  tf_x_points = tf.placeholder(tf.float32)
  line_eq = tf_get_line_eq(tf_x_points, tf_y_points)

gauss_points = np.random.normal(0, 1, 1000)
hist, bin_edges = np.histogram(gauss_points, bins=50)
hist = hist/np.sum(hist)
bin_edges = bin_edges[:-1]

x_points = [bin_edges[0], bin_edges[1]]
y_points = [hist[0], hist[1]]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config=config, graph=graph) as sess:
    sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
    t0 = ti.time()
    m,b = sess.run(line_eq, feed_dict={tf_x_points:x_points, tf_y_points:y_points})
    t1 = ti.time()
    tf_tot = t1-t0
    print("tf_time: ", tf_tot)

    t0 = ti.time()
    m2,b2 = np_get_line_eq(x_points, y_points)
    t1 = ti.time()
    np_tot = t1-t0
    print("np_time: ", np_tot)
    print("time difference (np-tf): ", np_tot-tf_tot)
