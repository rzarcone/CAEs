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

file_location = "/home/dpaiton/Work/Datasets/imagenet/imgs.txt"
num_gpus = 1
num_threads = 5
num_epochs = 300
shuffle = True
seed = 1234567890

num_outputs = 800
batch_size = 100
patch_size_y = 16
patch_size_x = 16
stride_y = 2
stride_x = 2
img_shape_y = 256
img_shape_x = 256
num_colors = 3

min_after_dequeue = 10  
num_read_threads = num_threads * num_gpus
capacity = min_after_dequeue + (num_read_threads + 1) * batch_size

w_shape = [patch_size_y, patch_size_x, num_colors, num_outputs]
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
filename_queue = tf.train.string_input_producer(filenames, num_epochs, shuffle=True, seed=seed,
  capacity=capacity)

# FIFO queue requires that all images have the same dtype & shape
queue = tf.FIFOQueue(capacity, dtypes=[tf.float32], shapes=[256, 256, 3])
# Enqueues one element at a time
enqueue_op = queue.enqueue(read_image(filename_queue)) 
# Reads a batch of images from the queue
x = queue.dequeue_many(batch_size) # Requires that all images are the same shape

# Holds a list of enqueue operations for a queue, each to be run in a thread.
qr = tf.train.QueueRunner(queue, [enqueue_op] * num_read_threads)

global_step = tf.Variable(0, trainable=False, name="global_step")

w = tf.get_variable(name="w", dtype=tf.float32,
  initializer=tf.truncated_normal(w_shape, mean=0.0, stddev=1.0, dtype=tf.float32,
  name="phi_init"), trainable=True)
u = tf.nn.conv2d(x, w, [1, stride_y, stride_x, 1], padding="SAME", name="activation")
x_ = tf.nn.conv2d_transpose(tf.nn.relu(u), w, tf.shape(x), [1, sride_y, stride_x, 1],
  padding="SAME", name="reconstruction")
loss = tf.mean(0.5 * tf.sum(tf.pow(tf.sub(x, x_), 2.0), reduction_indices=[1, 2, 3]))
train_op = tf.train.GradientDescentOptimizer.minimize(loss, global_step=global_step)

# Must initialize local variables as well as global to init num_epochs
# in tf.train.string_input_producer
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
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
        #data = sess.run(x)
        sess.run(train_op)
        step = sess.run(global_step)
        if step % 10 == 0:
          loss = sess.run(loss)
          print("step %g loss %g"%(step, loss))
  except tf.errors.OutOfRangeError:
    print("Done training -- epoch limit reached")
  finally:
    coord.request_stop()
  coord.join(enqueue_threads)
