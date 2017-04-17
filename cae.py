import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
import utils.plot_functions as pf
import numpy as np

"""Function to preprocess a single image"""
def preprocess_image(image):
  cropped_image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
  cropped_image = tf.to_float(cropped_image, name="ToFlaot")
  cropped_image = tf.div(cropped_image, 255.0)
  cropped_image = tf.subtract(cropped_image, tf.reduce_mean(cropped_image))
  ## grayscale ## 
  #cropped_image = tf.image.rgb_to_grayscale(cropped_image)
  return cropped_image

"""Function to load in a single image"""
def read_image(filename_queue):
  # Read an entire image file at once
  image_reader = tf.WholeFileReader()
  filename, image_file = image_reader.read(filename_queue)
  # If the image has 1 channel (grayscale) it will broadcast to 3
  image = tf.image.decode_jpeg(image_file, channels=3)
  # We want all images to be of the same size
  cropped_image = preprocess_image(image)
  return cropped_image

"""Devisive normalizeation nonlinearity"""
def gdn(layer_num, u_in, inverse):
  u_in_shape = u_in.get_shape()
  with tf.name_scope("gdn"+str(layer_num)) as scope:
    #small_const = tf.multiply(tf.ones(u_in_shape[3]),2e-5)
    small_const = tf.multiply(tf.ones(u_in_shape[3]),1e-3)
    b_gdn = tf.Variable(tf.add(tf.zeros(u_in_shape[3]), small_const), 
      trainable=True, dtype=tf.float32,  name="gdn_bias"+str(layer_num))
    b_threshold = tf.where(tf.less(b_gdn, tf.constant(1e-3, dtype=tf.float32)),
      tf.multiply(tf.ones_like(b_gdn), 1e-3), b_gdn)
    w_gdn_shape = [u_in_shape[3]]*2
    w_gdn_init = tf.multiply(tf.ones(shape=w_gdn_shape, dtype=tf.float32), 1e-3)
    w_gdn = tf.Variable(w_gdn_init, trainable=True, name="w_gdn"+str(layer_num))
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

  with tf.name_scope("biases"+str(layer_num)) as scope:
    if decode:
      b = tf.Variable(tf.zeros(w_shape[2]), trainable=True,
        name="latent_bias"+str(layer_num)) 
    else:
      b = tf.Variable(tf.zeros(w_shape[3]), trainable=True,
        name="latent_bias"+str(layer_num)) 

  with tf.name_scope("hidden"+str(layer_num)) as scope:
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

"""Print shapes of all u in u_list"""
def u_print(u_list):
  u_print_str = ""
  for idx, u in enumerate(u_list):
   u_eval = sess.run(u)
   u_shape = u_eval.shape
   num_u = u_eval.size
   u_print_str+="\tu"+str(idx)+"_shape: "+str(u_shape)+"\t"+str(num_u)+"\n"
   print(u_print_str)

#general params
file_location = "/home/dpaiton/Work/Datasets/imagenet/imgs.txt"
num_gpus = 1
gpu_id = "/gpu:0"
num_threads = 5
num_epochs = 10
seed = 1234567890

#image params
shuffle = True
batch_size = 100
img_shape_y = 256
img_shape_x = 256
num_colors = 3
batches_per_epoch = 450

#learning rates
init_learning_rate = 1e-4
decay_steps = 600 #0.5*batch_size*batches_per_epoch
staircase = True
decay_rate = 0.1 # for ADADELTA

#layer params
god_damn_network = True
relu = False
input_channels = [num_colors,192,192]#[num_colors, 128, 128]
output_channels = [192,192,192]#[128, 128, 128]
patch_size_y = [9,5,5]#[9, 5, 5]
strides = [4,2,2]#[4, 2, 2]

#queue params
patch_size_x = patch_size_y
w_shapes = [vals for vals in zip(patch_size_y, patch_size_x, input_channels,
  output_channels)]

#decoding is inverse of encoding
w_shapes += w_shapes[::-1]
input_channels += input_channels[::-1]
output_channels += output_channels[::-1]
strides += strides[::-1]
num_layers = len(w_shapes)

min_after_dequeue = 10

im_shape = [img_shape_y, img_shape_x, num_colors]
num_read_threads = num_threads * num_gpus
capacity = min_after_dequeue + (num_read_threads + 1) * batch_size

graph = tf.Graph()
with tf.device(gpu_id):
  with graph.as_default():
    with tf.name_scope("queue") as scope:
      # Make a list of filenames. Strip removes "\n" at the end
      # file_location contains image locations separated by newlines (piped from ls)
      filenames = tf.constant([string.strip()
        for string
        in open(file_location, "r").readlines()])

      # Turn list of filenames into a string producer to feed names for each thread
      # Shuffling happens here - should be faster than shuffling after the images are loaded
      # Capacity is the max capacity of the queue - can be adjusted as needed
      filename_queue = tf.train.string_input_producer(filenames, num_epochs, shuffle=shuffle,
        seed=seed, capacity=capacity)

      # FIFO queue requires that all images have the same dtype & shape
      queue = tf.FIFOQueue(capacity, dtypes=[tf.float32], shapes=im_shape)
      # Enqueues one element at a time
      enqueue_op = queue.enqueue(read_image(filename_queue)) 

    with tf.name_scope("input") as scope:
      # Reads a batch of images from the queue
      x = queue.dequeue_many(batch_size) # Requires that all images are the same shape

    with tf.name_scope("queue") as scope:
      # Holds a list of enqueue operations for a queue, each to be run in a thread.
      qr = tf.train.QueueRunner(queue, [enqueue_op] * num_read_threads)

    with tf.name_scope("step_counter") as scope:
      global_step = tf.Variable(0, trainable=False, name="global_step")

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
      w_list.append(w)
      u_list.append(u_out)
      b_list.append(b)
      b_gdn_list.append(b_gdn)
      w_gdn_list.append(w_gdn)

    with tf.name_scope("loss") as scope:
      total_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(u_list[0], u_list[-1]), 2.0),
        reduction_indices=[1,2,3]))

    with tf.name_scope("optimizers") as scope:
      learning_rates = tf.train.exponential_decay(
        learning_rate=init_learning_rate,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase,
        name="annealing_schedule")
      #optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=decay_rate, name="grad_optimizer")
      train_vars = w_list + b_list
      train_vars += b_gdn_list + w_gdn_list if god_damn_network else []
      optimizer = tf.train.AdamOptimizer(learning_rates, name="grad_optimizer")
      #optimizer = tf.train.GradientDescentOptimizer(learning_rates, name="grad_optimizer")
      grads_and_vars = optimizer.compute_gradients(total_loss, var_list=train_vars)
      train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    with tf.name_scope("performance_metrics") as scope:
      MSE = tf.reduce_mean(tf.pow(tf.subtract(u_list[0], u_list[-1]), 2.0), name="mean_squared_error")
      pSNRdB = tf.multiply(10.0, tf.log(tf.div(tf.pow(1.0, 2.0), MSE)), name="recon_quality")

    with tf.name_scope("summaries") as scope:
      tf.summary.image("input", u_list[0])
      tf.summary.image("reconstruction",u_list[-1])
      [tf.summary.histogram("u"+str(idx),u) for idx,u in enumerate(u_list)]
      [tf.summary.histogram("w"+str(idx),w) for idx,w in enumerate(w_list)]
      [tf.summary.histogram("b"+str(idx),b) for idx,b in enumerate(b_list)]
      [tf.summary.histogram("b_gdn"+str(idx),u) for idx,u in enumerate(b_gdn_list)]
      [tf.summary.histogram("w_gdn"+str(idx),w) for idx,w in enumerate(w_gdn_list)]
      tf.summary.scalar("total_loss", total_loss)
      tf.summary.scalar("MSE", MSE)
      tf.summary.scalar("pSNRdB", pSNRdB)

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=0.5
    config.gpu_options.allow_growth = True
    config.log_device_placement = False # for debugging - log devices used by each variable

    # Must initialize local variables as well as global to init num_epochs
    # in tf.train.string_input_produce
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("/home/rzarcone/CAE_Project/CAEs" + "/train", graph)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

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
      sess.run(train_op)
      step = sess.run(global_step)
      if step % 10 == 0:
        ### SUMMARY ##
        summary = sess.run(merged)
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
        eval_total_loss = sess.run(total_loss)
        snr = sess.run(pSNRdB)
        print("step %g\tloss %g\tpSNRdB %g"%(step, eval_total_loss, snr))
        #u_print(u_list)
  coord.request_stop()
  coord.join(enqueue_threads)
