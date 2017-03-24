import tensorflow as tf

"""Function to preprocess a single image"""
def preprocess_image(image):
  cropped_image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
  cropped_image = tf.to_float(cropped_image, name='ToFlaot')
  cropped_image = tf.div(cropped_image, 255.0)
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
learning_rate = 1e-4
num_outputs = [128] 
patch_size_y = [8]
stride_y = [2]

#queue params
min_after_dequeue = 10  

assert ((img_shape_y - patch_size_y[0]) % stride_y[0] == 0), (
  "Patch & Stride must divide evenly into the image")

patch_size_x = patch_size_y
stride_x = stride_y
im_shape = [img_shape_y, img_shape_x, num_colors]
num_read_threads = num_threads * num_gpus
capacity = min_after_dequeue + (num_read_threads + 1) * batch_size

w_shapes = [[py, px, 3, no] for py in patch_size_y for px in patch_size_x for no in num_outputs]

graph = tf.Graph()
with graph.as_default():
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
  # Reads a batch of images from the queue
  x = queue.dequeue_many(batch_size) # Requires that all images are the same shape

  # Holds a list of enqueue operations for a queue, each to be run in a thread.
  qr = tf.train.QueueRunner(queue, [enqueue_op] * num_read_threads)

  global_step = tf.Variable(0, trainable=False, name="global_step")

  w_enc = tf.get_variable(name="w_enc", dtype=tf.float32,
    initializer=tf.truncated_normal(w_shapes[0], mean=0.0, stddev=1.0, dtype=tf.float32,
    name="w_enc_init"), trainable=True)
  u = tf.nn.relu(tf.nn.conv2d(x, w_enc, [1, stride_y[0], stride_x[0], 1], padding="SAME",
    use_cudnn_on_gpu=True, name="activation"))
  w_dec = tf.get_variable(name="w_dec", dtype=tf.float32,
    initializer=tf.truncated_normal(w_shapes[0], mean=0.0, stddev=1.0, dtype=tf.float32,
    name="w_dec_init"), trainable=True)
  xh_ = tf.nn.conv2d_transpose(u, w_dec, [batch_size]+im_shape, [1, stride_y[0], stride_x[0], 1],
    padding="SAME", name="reconstruction")
  total_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(x,xh_), 2.0),
    reduction_indices=[1,2,3]))

  optimizer = tf.train.GradientDescentOptimizer(learning_rate, name="grad_optimizer")
  grads_and_vars = optimizer.compute_gradients(total_loss, var_list=[w_enc, w_dec])
  train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

  MSE = tf.reduce_mean(tf.pow(tf.subtract(x, xh_), 2.0), name="mean_squared_error")
  pSNRdB = tf.multiply(10.0, tf.log(tf.div(tf.pow(1.0, 2.0), MSE)), name="recon_quality")

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  # Must initialize local variables as well as global to init num_epochs
  # in tf.train.string_input_producer
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
          eval_total_loss = sess.run(total_loss)
          snr = sess.run(pSNRdB)
          print("step %g\tloss %g\tpSNRdB %g"%(step, eval_total_loss, snr))
  except tf.errors.OutOfRangeError:
    print("Done training -- epoch limit reached")
  finally:
    coord.request_stop()
  coord.join(enqueue_threads)
