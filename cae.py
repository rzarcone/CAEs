import matplotlib
matplotlib.use("Agg")

import os
import tensorflow as tf
import utils.plot_functions as pf
import numpy as np
import utils.get_data as get_data
import utils.mem_utils as mem_utils

"""Function to preprocess a single image"""
def preprocess_image(image):
  # We want all images to be of the same size
  cropped_image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
  cropped_image = tf.to_float(cropped_image, name="ToFlaot")
  cropped_image = tf.div(cropped_image, 255.0)
  cropped_image = tf.subtract(cropped_image, tf.reduce_mean(cropped_image))
  cropped_image = tf.image.rgb_to_grayscale(cropped_image)
  return cropped_image

"""Function to load in a single image"""
def read_image(filename_queue):
  # Read an entire image file at once
  image_reader = tf.WholeFileReader()
  filename, image_file = image_reader.read(filename_queue)
  # If the image has 1 channel (grayscale) it will broadcast to 3
  image = tf.image.decode_jpeg(image_file, channels=3)
  cropped_image = preprocess_image(image)
  return cropped_image

def memristorize(u_in, memristor_std_eps):
  with tf.variable_scope("memristor_transform") as scope:
    path = 'data/Partial_Reset_PCM.pkl'
    #n_mem = tf.reduce_prod(u_out.get_shape()[1:])
    #n_mem = 32768 # 49152 for color, 32768 for grayscale
    (vs_data, mus_data, sigs_data,
      orig_VMIN, orig_VMAX, orig_RMIN,
      orig_RMAX) = get_data.get_memristor_data(path, n_mem, num_ext=5,
      norm_min=mem_v_min, norm_max=mem_v_max)
    v_clip = tf.clip_by_value(u_in, clip_value_min=mem_v_min, clip_value_max=mem_v_max)
    #v_trans = tensor_scaler(v_clip,orig_VMIN,orig_VMAX)
    r = mem_utils.memristor_output(v_clip, memristor_std_eps, vs_data, mus_data, sigs_data,
      interp_width=np.array(vs_data[1, 0] - vs_data[0, 0]).astype('float32'))
    #r_trans =  tensor_scaler(r, orig_RMIN, orig_RMAX)
    return tf.reshape(r, shape=u_in.get_shape(), name="mem_r")

"""Devisive normalizeation nonlinearity"""
def gdn(layer_num, u_in, inverse):
  u_in_shape = u_in.get_shape()
  with tf.variable_scope("gdn"+str(layer_num)) as scope:
    #small_const = tf.multiply(tf.ones(u_in_shape[3]),2e-5)
    small_const = tf.multiply(tf.ones(u_in_shape[3]),1e-3)
    b_gdn = tf.get_variable(name="gdn_bias"+str(layer_num), dtype=tf.float32,
      initializer=tf.add(tf.zeros(u_in_shape[3]), small_const), trainable=True)
    b_threshold = tf.where(tf.less(b_gdn, tf.constant(1e-3, dtype=tf.float32)),
      tf.multiply(tf.ones_like(b_gdn), 1e-3), b_gdn)
    w_gdn_shape = [u_in_shape[3]]*2
    w_gdn_init = tf.multiply(tf.ones(shape=w_gdn_shape, dtype=tf.float32), 1e-3)
    w_gdn = tf.get_variable(name="w_gdn"+str(layer_num), dtype=tf.float32,
      initializer=w_gdn_init, trainable=True)
    w_threshold = tf.where(tf.less(w_gdn, tf.constant(1e-3, dtype=tf.float32)), w_gdn_init, w_gdn)
    collapsed_u_sq = tf.reshape(tf.square(u_in),
      shape=tf.stack([u_in_shape[0]*u_in_shape[1]*u_in_shape[2], u_in_shape[3]]))
    weighted_norm = tf.reshape(tf.matmul(collapsed_u_sq,tf.add(w_threshold,
      tf.transpose(w_threshold))), shape=tf.stack([u_in_shape[0], u_in_shape[1], u_in_shape[2],
      u_in_shape[3]]))
    GDN_const = tf.sqrt(tf.add(weighted_norm, b_threshold))
    if inverse:
      u_out = tf.multiply(u_in, GDN_const, name="u_gdn"+str(layer_num))
    else:
      u_out = tf.where(tf.less(GDN_const, tf.constant(1e-7, dtype=tf.float32)), u_in,
        tf.divide(u_in, GDN_const), name="u_gdn"+str(layer_num))
  return u_out, b_gdn, w_gdn

"""
Make layer that does activation(conv(u,w)+b)
  where activation() is either relu or GDN
"""
def layer_maker(layer_num, u_in, w_shape, w_init, stride, decode, relu, god_damn_network):
  b_gdn = None
  w_gdn = None
  with tf.variable_scope("weights"+str(layer_num)):
    w = tf.get_variable(name="w"+str(layer_num), shape=w_shape, dtype=tf.float32,
      initializer=w_init, trainable=True)

  with tf.variable_scope("biases"+str(layer_num)) as scope:
    if decode:
      b = tf.get_variable(name="latent_bias"+str(layer_num), dtype=tf.float32,
        initializer=tf.zeros(w_shape[2]), trainable=True)
    else:
      b = tf.get_variable(name="latent_bias"+str(layer_num), dtype=tf.float32,
        initializer=tf.zeros(w_shape[3]), trainable=True)

  with tf.variable_scope("hidden"+str(layer_num)) as scope:
    if decode:
      if god_damn_network:
        u_in, b_gdn, w_gdn = gdn(layer_num, u_in, decode)
      height_const = 0 if u_in.get_shape()[1] % stride == 0 else 1
      out_height = (u_in.get_shape()[1] * stride) - height_const
      width_const = 0 if u_in.get_shape()[2] % stride == 0 else 1
      out_width = (u_in.get_shape()[2] * stride) - width_const
      out_shape = tf.stack([u_in.get_shape()[0], # Batch
        out_height, # Height
        out_width, # Width
        tf.constant(w_shape[2], dtype=tf.int32)]) # Channels
      u_out = tf.add(tf.nn.conv2d_transpose(u_in, w, out_shape,
        strides=[1, stride, stride, 1], padding="SAME"), b,
        name="activation"+str(layer_num))
      if relu:
        u_out = tf.nn.relu(u_out, name="relu_activation")
    else:
      u_out = tf.add(tf.nn.conv2d(u_in, w, [1, stride, stride, 1],
        padding="SAME", use_cudnn_on_gpu=True), b, name="activation"+str(layer_num))
      if relu:
        u_out = tf.nn.relu(u_out, name="relu_activation")
      if god_damn_network:
        u_out, b_gdn, w_gdn = gdn(layer_num, u_out, decode)
  return w, b, u_out, b_gdn, w_gdn

"""
Average gradients computed per GPU
Args:
 grad_list: list of lists of (gradint, variable) tuples.
   Outer list is per tower, inner list is per varaible
"""
def average_gradients(grad_list):
  avg_grads = []
  for grad_and_vars in zip(*grad_list):
    # star unpacks outer loop, such that each grad_and_vars
    # looks like the following:
    #   iter 0: ((grad0_gpu0, var0_gpu0), .., (grad0_gpuN, var0_gpuN))
    #   iter 1: ((grad1_gpu0, var1_gpu0), .., (grad1_gpuN, var1_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.reduce_mean(tf.concat(axis=0, values=grads), axis=0)
    v = grad_and_vars[0][1] # the variables are shared across towers
    avg_grads.append((grad, v))
  return avg_grads

"""Print shapes of all u in u_list"""
def u_print(u_list):
  u_print_str = ""
  for idx, u in enumerate(u_list):
   u_eval = sess.run(u)
   u_shape = u_eval.shape
   num_u = u_eval.size
   u_print_str+="\tu"+str(idx)+"_shape: "+str(u_shape)+"\t"+str(num_u)+"\n"
   print(u_print_str)

#shitty hard coding
n_mem = 32768 # 49152 for color, 32768 for grayscale

#general params
#file_location = "/media/tbell/datasets/natural_images.txt"
#file_location = "/media/tbell/datasets/imagenet/imgs.txt"
file_location = "/media/tbell/datasets/flickr_yfcc100m/flickr_images.txt"
gpu_ids = ["0", "1"]
output_location = os.path.expanduser("~")+"/CAE_Project/CAEs/train/"
num_threads = 5
num_epochs = 2
epoch_size = 7e4
eval_interval = 10
seed = 1234567890

#image params
shuffle_inputs = True
batch_size = 25
img_shape_y = 256
num_colors = 1

#learning rates
init_learning_rate = 5e-4
decay_steps = 1000 #0.5*epoch_size
staircase = True
decay_rate = 0.9 # for ADADELTA

#layer params
memristorify = True
god_damn_network = True 
relu = False 
input_channels = [num_colors, 128, 128] #192 for color
output_channels = [128, 128, 128]
patch_size_y = [9, 5, 5]#[3,3,5,5]
strides = [4, 2, 2]#[2,2,2,2]
GAMMA = 1.0  # slope of the out of bounds cost
mem_v_min = -1.0
mem_v_max = 1.0

#queue params
num_gpus = len(gpu_ids)
patch_size_x = patch_size_y
effective_batch_size = num_gpus*batch_size
batches_per_epoch = int(np.floor(epoch_size/effective_batch_size))
w_shapes = [vals for vals in zip(patch_size_y, patch_size_x, input_channels,
  output_channels)]

#decoding is inverse of encoding
w_shapes += w_shapes[::-1]
input_channels += input_channels[::-1]
output_channels += output_channels[::-1]
strides += strides[::-1]
num_layers = len(w_shapes)
min_after_dequeue = 20
img_shape_x = img_shape_y
im_shape = [img_shape_y, img_shape_x, num_colors]
num_read_threads = num_threads# * num_gpus #TODO:only running on cpu - so don't mult by num_gpus?
capacity = min_after_dequeue + (num_read_threads + 1) * batch_size

graph = tf.Graph()
with graph.as_default(),tf.device('/cpu:0'):
  with tf.name_scope("step_counter") as scope:
    global_step = tf.Variable(0, trainable=False, name="global_step")

  with tf.variable_scope("optimizers") as scope:
    learning_rates = tf.train.exponential_decay(
      learning_rate=init_learning_rate,
      global_step=global_step,
      decay_steps=decay_steps,
      decay_rate=decay_rate,
      staircase=staircase,
      name="annealing_schedule")
    optimizer = tf.train.AdamOptimizer(learning_rates, name="grad_optimizer")

  with tf.variable_scope("placeholders") as scope:
    #n_mem = tf.reduce_prod(u_in.get_shape()[1:])
    #n_mem = 32768 # 49152 for color, 32768 for grayscale
    memristor_std_eps = tf.placeholder(tf.float32, shape=(effective_batch_size, n_mem))

  with tf.variable_scope("queue") as scope:
    # Make a list of filenames. Strip removes "\n" at the end
    # file_location contains image locations separated by newlines (piped from ls)
    filenames = tf.constant([string.strip()
      for string
      in open(file_location, "r").readlines()])

    # Turn list of filenames into a string producer to feed names for each thread
    # Shuffling happens here - should be faster than shuffling after the images are loaded
    # Capacity is the max capacity of the queue - can be adjusted as needed
    filename_queue = tf.train.string_input_producer(filenames, num_epochs,
      shuffle=shuffle_inputs, seed=seed, capacity=capacity)

    # FIFO queue requires that all images have the same dtype & shape
    queue = tf.FIFOQueue(capacity, dtypes=[tf.float32], shapes=im_shape)
    # Enqueues one element at a time
    enqueue_op = queue.enqueue(read_image(filename_queue))

  with tf.variable_scope("queue") as scope:
    # Holds a list of enqueue operations for a queue, each to be run in a thread.
    qr = tf.train.QueueRunner(queue, [enqueue_op] * num_read_threads)

  with tf.variable_scope("input") as scope:
    # Reads a batch of images from the queue
    x = queue.dequeue_many(batch_size) # Requires that all images are the same shape

  gradient_list = []
  # tf.get_variable_scope().reuse_variables() call will only apply
  # to items within this with statement
  with tf.variable_scope(tf.get_variable_scope()):
    for gpu_id in gpu_ids:
      with tf.device("/gpu:"+gpu_id):
        with tf.name_scope("tower_"+gpu_id) as scope:
          w_list = []
          u_list = [x]
          b_list = []
          b_gdn_list = []
          w_gdn_list = []
          w_inits = [tf.contrib.layers.xavier_initializer_conv2d(uniform=False, seed=seed,
            dtype=tf.float32) for _ in np.arange(num_layers/2)]
          w_inits += w_inits # decode inits are the same as encode inits
          for layer_idx, w_shapes_strides in enumerate(zip(w_shapes, strides)):
            decode = False if layer_idx < num_layers/2 else True
            w, b, u_out, b_gdn, w_gdn = layer_maker(layer_idx, u_list[layer_idx], w_shapes_strides[0],
              w_inits[layer_idx], w_shapes_strides[1], decode, relu, god_damn_network)
            if memristorify:
              if layer_idx == num_layers/2-1:
                with tf.variable_scope("loss") as scope:
                  # Penalty for going out of bounds
                  reg_loss = tf.reduce_mean(tf.reduce_sum(GAMMA * (tf.nn.relu(u_out- mem_v_max)
                    + tf.nn.relu(mem_v_min - u_out)), axis=[1,2,3]))
                memristor_std_eps_slice = tf.split(value=memristor_std_eps,
                  num_or_size_splits=num_gpus, axis=0)[int(gpu_id)]
                u_out = memristorize(u_out, memristor_std_eps_slice)
            w_list.append(w)
            u_list.append(u_out)
            b_list.append(b)
            b_gdn_list.append(b_gdn)
            w_gdn_list.append(w_gdn)

          with tf.variable_scope("loss") as scope:
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(u_list[0], u_list[-1]), 2.0),
              axis=[1,2,3]))
            total_loss = tf.add_n([recon_loss, reg_loss], name="total_loss")

          with tf.variable_scope("optimizers") as scope:
            train_vars = w_list + b_list
            train_vars += b_gdn_list + w_gdn_list if god_damn_network else []
            grads_and_vars = optimizer.compute_gradients(total_loss, var_list=train_vars)
            gradient_list.append(grads_and_vars)

          tf.get_variable_scope().reuse_variables()

  avg_grads = average_gradients(gradient_list)
  with tf.variable_scope("optimizers") as scope:
    train_op = optimizer.apply_gradients(avg_grads, global_step=global_step)

  with tf.name_scope("savers") as scope:
    full_saver = tf.train.Saver(var_list=train_vars, max_to_keep=2)

  with tf.name_scope("performance_metrics") as scope:
    MSE = tf.reduce_mean(tf.square(tf.subtract(u_list[0],
      tf.clip_by_value(u_list[-1], clip_value_min=-1.0, clip_value_max=1.0))),
      name="mean_squared_error")
    SNRdB = tf.multiply(10.0, tf.log(tf.div(tf.square(tf.nn.moments(u_list[0], axes=[0,1,2,3])[1]), MSE)), name="recon_quality")

  with tf.name_scope("summaries") as scope:
    tf.summary.image("input", u_list[0])
    tf.summary.image("reconstruction",u_list[-1])
    [tf.summary.histogram("u"+str(idx),u) for idx,u in enumerate(u_list)]
    [tf.summary.histogram("w"+str(idx),w) for idx,w in enumerate(w_list)]
    [tf.summary.histogram("b"+str(idx),b) for idx,b in enumerate(b_list)]
    if god_damn_network:
      [tf.summary.histogram("b_gdn"+str(idx),u) for idx,u in enumerate(b_gdn_list)]
      [tf.summary.histogram("w_gdn"+str(idx),w) for idx,w in enumerate(w_gdn_list)]
    tf.summary.scalar("total_loss", total_loss)
    tf.summary.scalar("MSE", MSE)
    tf.summary.scalar("SNRdB", SNRdB)

  # Must initialize local variables as well as global to init num_epochs
  # in tf.train.string_input_produce
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter("/home/rzarcone/CAE_Project/CAEs" + "/train", graph)
  init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction=0.5
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False # for debugging - log devices used by each variable
with tf.Session(config=config, graph=graph) as sess:
  sess.run(init_op)
  # Coordinator manages threads, checks for stopping requests
  for n in range(num_epochs):
    coord = tf.train.Coordinator()
    # Both start_queue_runners and create_threads must be called to enqueue the images
    # TODO: Why do we need to do both of these? the start=True flag on create_threads should cover
    #        what start_queue_runners does.
    #       Hard-coded batches_per_epoch must be less than the actual number of available batches
    #        how to allow the queue to totally empty before refilling?
    _ = tf.train.start_queue_runners(sess, coord, start=True)
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    for i in range(batches_per_epoch):
      #n_mem_eval = sess.run(tf.to_int32(tf.size(u_list[int(num_layers/2)])/u_list[int(num_layers/2)].get_shape()[0]))
      #n_mem_eval =49152 # 49152
      mem_std_eps = np.random.standard_normal((effective_batch_size, n_mem)).astype(np.float32)
      feed_dict={memristor_std_eps:mem_std_eps}
      sess.run(train_op, feed_dict=feed_dict)
      step = sess.run(global_step, feed_dict=feed_dict)
      if step % eval_interval == 0:
        ### SUMMARY ##
        summary = sess.run(merged, feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        # TODO: Verify that save_data_tiled correctly saves color weights
        #weight_filename = "/home/rzarcone/CAE_Project/CAEs/train/weights/"
        #w_enc_eval = np.squeeze(sess.run(tf.transpose(w_list[0], perm=[3,0,1,2])))
        #pf.save_data_tiled(w_enc_eval, normalize=True, title="Weights0",
        #  save_filename=weight_filename+"Weights_enc.png")
        #w_dec_eval = np.squeeze(sess.run(tf.transpose(w_list[-1], perm=[3,0,1,2])))
        #pf.save_data_tiled(w_dec_eval, normalize=True, title="Weights-1",
        #  save_filename=weight_filename+"Weights_dec.png")
        ## Print stuff
        [ev_reg_loss, ev_recon_loss, ev_total_loss] = sess.run([reg_loss, recon_loss, total_loss],
          feed_dict=feed_dict)
        snr = sess.run(MSE, feed_dict=feed_dict)
        print("step %04d\treg_loss %03g\trecon_loss %g\ttotal_loss %g\tMSE %g"%(
          step, ev_reg_loss, ev_recon_loss, ev_total_loss, snr))
        #u_print(u_list)
    if not os.path.exists(output_location+"checkpoints/"):
      os.makedirs(output_location+"checkpoints/")
    full_saver.save(sess, save_path=output_location+"/checkpoints/chkpt_", global_step=global_step)
  coord.request_stop()
  coord.join(enqueue_threads)
