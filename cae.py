import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import utils.plot_functions as pf
import numpy as np

"""Function to preprocess a single image"""
def preprocess_image(image):
  cropped_image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
  cropped_image = tf.to_float(cropped_image, name='ToFlaot')
  cropped_image = tf.div(cropped_image, 255.0)
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

#general params
file_location = "/home/dpaiton/Work/Datasets/imagenet/imgs.txt"
num_gpus = 1
num_threads = 5
num_epochs = 1
seed = 1234567890

#image params
shuffle = True
num_epochs = 1
batch_size = 100
img_shape_y = 256
img_shape_x = 256
num_colors = 3

#layer params
learning_rate = 1e-8
decay_rate = 0.95 # for ADADELTA
num_input_channels = [num_colors, 32]
num_outputs = [32, 8] 
patch_size_y = [3, 3]
stride = [2, 2]

#queue params
min_after_dequeue = 10  

patch_size_x = patch_size_y
im_shape = [img_shape_y, img_shape_x, num_colors]
num_read_threads = num_threads * num_gpus
capacity = min_after_dequeue + (num_read_threads + 1) * batch_size

w_shapes = [vals for vals in zip(patch_size_y, patch_size_x, num_input_channels, num_outputs)]

graph = tf.Graph()
with graph.as_default():
  with tf.name_scope("queue") as scope:
    # Make a list of filenames. Strip removes '\n' at the end
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

  with tf.name_scope("weights1") as scope:
    w_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False, seed=seed,
      dtype=tf.float32)
    w_enc = tf.get_variable(name="w_enc", shape=w_shapes[0], dtype=tf.float32,
      initializer=w_init, trainable=True)
    w_dec = tf.get_variable(name="w_dec", shape=w_shapes[0], dtype=tf.float32,
      initializer=w_init, trainable=True)

  with tf.name_scope("weights2") as scope:
    w_init2 = tf.contrib.layers.xavier_initializer_conv2d(uniform=False, seed=seed,
      dtype=tf.float32)
    w_enc2 = tf.get_variable(name="w_enc2", shape=w_shapes[1], dtype=tf.float32,
      initializer=w_init2, trainable=True)
    w_dec2 = tf.get_variable(name="w_dec2", shape=w_shapes[1], dtype=tf.float32,
      initializer=w_init2, trainable=True)

  with tf.name_scope('biases') as scope:
    b_hidden = tf.Variable(tf.zeros(num_outputs[0]), trainable=True, name="latent_bias")

  with tf.name_scope('biases2') as scope:
    b_hidden2 = tf.Variable(tf.zeros(num_outputs[1]), trainable=True, name="latent_bias")

  with tf.name_scope("hidden") as scope:
    u = tf.nn.relu(tf.add(tf.nn.conv2d(x, w_enc, [1, stride[0], stride[0], 1], padding="SAME",
      use_cudnn_on_gpu=True, name="activation"), b_hidden))

  with tf.name_scope("hidden2") as scope:
    u2 = tf.nn.relu(tf.add(tf.nn.conv2d(u, w_enc2, [1, stride[1], stride[1], 1], padding="SAME",
      use_cudnn_on_gpu=True, name="activation"), b_hidden2))

  with tf.name_scope("deconv") as scope:
    uh_ = tf.nn.relu(tf.nn.conv2d_transpose(u2, w_dec2, tf.shape(u),
      strides=[1, stride[1], stride[1], 1], padding="SAME", name="reconstruction"))

  with tf.name_scope("reconstruction") as scope:
    xh_ = tf.nn.relu(tf.nn.conv2d_transpose(uh_, w_dec, [batch_size]+im_shape,
      strides=[1, stride[0], stride[0], 1], padding="SAME", name="reconstruction"))

  with tf.name_scope("loss") as scope:
    total_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(x, xh_), 2.0),
      reduction_indices=[1,2,3]))

  with tf.name_scope("optimizers") as scope:
    #optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=decay_rate, name="grad_optimizer")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate, name="grad_optimizer")
    grads_and_vars = optimizer.compute_gradients(total_loss, var_list=[w_enc, w_enc2, w_dec, w_dec2, b_hidden, b_hidden2])
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

  with tf.name_scope("performance_metrics") as scope:
    MSE = tf.reduce_mean(tf.pow(tf.subtract(x, xh_), 2.0), name="mean_squared_error")
    pSNRdB = tf.multiply(10.0, tf.log(tf.div(tf.pow(1.0, 2.0), MSE)), name="recon_quality")

   
  with tf.name_scope("summaries") as scope:
    tf.summary.image('x', x) 
    tf.summary.image('xh',xh_)
    tf.summary.histogram('x', x) 
    tf.summary.histogram('xh',xh_)
    tf.summary.histogram('w_enc', w_enc) 
    tf.summary.histogram('w_dec', w_dec) 
    tf.summary.histogram('bias', b_hidden) 
    tf.summary.histogram('hidden_activations', u) 
    tf.summary.histogram('w_enc2', w_enc2) 
    tf.summary.histogram('w_dec2', w_dec2) 
    tf.summary.histogram('bias2', b_hidden2) 
    tf.summary.histogram('hidden_activations2', u2) 
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('MSE', MSE)
    tf.summary.scalar('pSNRdB', pSNRdB)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  # Must initialize local variables as well as global to init num_epochs
  # in tf.train.string_input_produce
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('/home/rzarcone/CAE_Project/CAEs' + '/train', graph)
  init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session(graph=graph) as sess:
  sess.run(init_op)
  # Coordinator manages threads, checks for stopping requests
  coord = tf.train.Coordinator()
  # Both start_queue_runners and create_threads must be called to enqueue the images
  # TODO: Why do we need to do both of these? the start=True flag on create_threads should cover
  #        what start_queue_runners does.
  _ = tf.train.start_queue_runners(sess, coord, start=True)
  enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

  try:
    with coord.stop_on_exception():
      while not coord.should_stop():
        sess.run(train_op)
        step = sess.run(global_step)
        if step % 10 == 0:
          ### SUMMARY ##
          summary = sess.run(merged)
          train_writer.add_summary(summary, step)
          weight_filename = "/home/rzarcone/CAE_Project/CAEs/train/weights/"
          w_enc_eval = np.squeeze(sess.run(tf.transpose(w_enc, perm=[3,0,1,2]))) 
          w_dec_eval = np.squeeze(sess.run(tf.transpose(w_dec, perm=[3,0,1,2]))) 
          pf.save_data_tiled(w_enc_eval,
            normalize=False, title="Encoding Weights", save_filename=weight_filename+"encode.png")
          pf.save_data_tiled(w_dec_eval,
            normalize=False, title="Decoding Weights", save_filename=weight_filename+"decode.png")
          ## Print stuff
          eval_total_loss = sess.run(total_loss)
          snr = sess.run(pSNRdB)
          print("step %g\tloss %g\tpSNRdB %g"%(step, eval_total_loss, snr))
          print("u shape %g\tu2 shape %g"%(sess.run(u).size/batch_size, sess.run(u2).size/batch_size))
        if step == 400:
          coord.request_stop()
  except tf.errors.OutOfRangeError:
    print("Done training -- epoch limit reached")
  finally:
    coord.request_stop()
  coord.join(enqueue_threads)
