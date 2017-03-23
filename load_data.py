import tensorflow as tf

"""Function to preprocess a single image"""
def preprocess_image(image):
  cropped_image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
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
num_epochs = 1
batch_size = 100
num_read_threads = 10
min_after_dequeue = 10  
seed = 1234
capacity = min_after_dequeue + (num_read_threads + 1) * batch_size

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
queue = tf.FIFOQueue(capacity, dtypes=[tf.uint8], shapes=[256, 256, 3])
# Enqueues one element at a time
enqueue_op = queue.enqueue(read_image(filename_queue)) 
# Reads a batch of images from the queue
images = queue.dequeue_many(batch_size) # Requires that all images are the same shape

# Holds a list of enqueue operations for a queue, each to be run in a thread.
qr = tf.train.QueueRunner(queue, [enqueue_op] * num_read_threads)

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
        data = sess.run(images)
  except tf.errors.OutOfRangeError:
    print("Done training -- epoch limit reached")
  finally:
    coord.request_stop()
  coord.join(enqueue_threads)
