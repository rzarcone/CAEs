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

"""Make layer that does relu(conv(u,w)+b)"""
def layer_maker(layer_num, u_in, w_shape, w_init, stride, transpose):
  with tf.variable_scope('weights'+str(layer_num)):
    w = tf.get_variable(name='w'+str(layer_num), shape=w_shape, dtype=tf.float32,
      initializer=w_init, trainable=True)

  with tf.name_scope('biases'+str(layer_num)) as scope:                                                             
    if not transpose:
      b = tf.Variable(tf.zeros(w_shape[3]), trainable=True,
        name="latent_bias"+str(layer_num)) 
    else:
      b = tf.Variable(tf.zeros(w_shape[2]), trainable=True,
        name="latent_bias"+str(layer_num)) 

  with tf.name_scope("hidden"+str(layer_num)) as scope:
    if not transpose:
      u_out = tf.nn.relu(tf.add(tf.nn.conv2d(u_in, w, [1, stride, stride, 1], 
        padding="SAME", use_cudnn_on_gpu=True), b), name="activation"+str(layer_num))
    else:
      height_const = tf.where(tf.equal(tf.mod(tf.shape(u_in)[1], stride), 0), 0, 1)
      out_height = tf.shape(u_in)[1] * stride - height_const
      width_const = tf.where(tf.equal(tf.mod(tf.shape(u_in)[2], stride), 0), 0, 1)
      out_width = tf.shape(u_in)[2] * stride - width_const
      out_shape = tf.stack([tf.cast(tf.shape(u_in)[0], tf.int32), # Batch
        tf.cast(out_height, tf.int32), # Height
        tf.cast(out_width, tf.int32), # Width
        tf.constant(w_shape[2], dtype=tf.int32)]) # Channels
      u_out = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(u_in, w, out_shape,
        strides=[1, stride, stride, 1], padding="SAME"), b),
        name="activation"+str(layer_num))

  return w, b, u_out

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
input_channels = [num_colors]#[num_colors, 128, 128]
output_channels = [128]#[128, 128, 128]
patch_size_y = [9]#[9, 5, 5]
strides = [2]#[4, 2, 2]

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

  w_list = []
  u_list = [x]
  b_list = []

  w_inits = [tf.contrib.layers.xavier_initializer_conv2d(uniform=False, seed=seed,
    dtype=tf.float32) for _ in np.arange(num_layers/2)]
  w_inits += w_inits # decode inits are the same as encode inits
 
  for layer_idx, vals in enumerate(zip(w_shapes, strides)):
    transpose = False if layer_idx < num_layers/2 else True
    w, b, u_out = layer_maker(layer_idx, u_list[layer_idx], vals[0],
      w_inits[layer_idx], vals[1], transpose)
    w_list.append(w)
    u_list.append(u_out)
    b_list.append(b)

  with tf.name_scope("loss") as scope:
    total_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(u_list[0], u_list[-1]), 2.0),
      reduction_indices=[1,2,3]))

  with tf.name_scope("optimizers") as scope:
    #optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=decay_rate, name="grad_optimizer")
    train_vars = w_list + b_list
    optimizer = tf.train.GradientDescentOptimizer(learning_rate, name="grad_optimizer")
    grads_and_vars = optimizer.compute_gradients(total_loss, var_list=train_vars)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

  with tf.name_scope("performance_metrics") as scope:
    MSE = tf.reduce_mean(tf.pow(tf.subtract(u_list[0], u_list[-1]), 2.0), name="mean_squared_error")
    pSNRdB = tf.multiply(10.0, tf.log(tf.div(tf.pow(1.0, 2.0), MSE)), name="recon_quality")

  with tf.name_scope("summaries") as scope:
    tf.summary.image('input', u_list[0]) 
    tf.summary.image('reconstruction',u_list[-1])
    [tf.summary.histogram('u'+str(idx),u) for idx,u in enumerate(u_list)]
    [tf.summary.histogram('w'+str(idx),w) for idx,w in enumerate(w_list)]
    [tf.summary.histogram('b'+str(idx),b) for idx,b in enumerate(b_list)]
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

with tf.Session(config=config, graph=graph) as sess:
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
        #w_shapes = [sess.run(w).shape for w in w_list]
        #u_shapes = [sess.run(u).shape for u in u_list]
        #b_shapes = [sess.run(b).shape for b in b_list]
        #import IPython; IPython.embed(); raise SystemExit
        sess.run(train_op)
        step = sess.run(global_step)
        if step % 10 == 0:
          ### SUMMARY ##
          summary = sess.run(merged)
          train_writer.add_summary(summary, step)
          weight_filename = "/home/rzarcone/CAE_Project/CAEs/train/weights/"
          w_enc_eval = np.squeeze(sess.run(tf.transpose(w_list[0], perm=[3,0,1,2]))) 
          pf.save_data_tiled(w_enc_eval, normalize=True, title="Weights0",
            save_filename=weight_filename+"Weights_enc.png")
          w_dec_eval = np.squeeze(sess.run(tf.transpose(w_list[-1], perm=[3,0,1,2]))) 
          pf.save_data_tiled(w_dec_eval, normalize=True, title="Weights-1",
            save_filename=weight_filename+"Weights_dec.png")
          ## Print stuff
          eval_total_loss = sess.run(total_loss)
          snr = sess.run(pSNRdB)
          print("step %g\t\tloss %g\tpSNRdB %g"%(step, eval_total_loss, snr))
          #u_print_str = ""
          #for idx, u in enumerate(u_list):
          #  u_eval = sess.run(u)
          #  u_shape = u_eval.shape
          #  num_u = u_eval.size
          #  u_print_str+="\tu"+str(idx)+"_shape: "+str(u_shape)+"\t"+str(num_u)+"\n"
          #print(u_print_str)
        if step == 400:
          coord.request_stop()
  except tf.errors.OutOfRangeError:
    print("Done training -- epoch limit reached")
  finally:
    coord.request_stop()
  coord.join(enqueue_threads)
