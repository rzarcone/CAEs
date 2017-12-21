import tensorflow as tf
import numpy as np
import os
import utils.get_data as get_data
import utils.mem_utils as mem_utils
import utils.entropy_funcs as ef

class cae(object):
    def __init__(self, params):
      params = self.add_params(params)
      self.params = params
      self.make_dirs()
      self.construct_graph()

    """Adds additional parameters that can be computed from given parameters"""
    def add_params(self, params):
      params["weight_save_filename"] = params["output_location"]+"/weights/"
      #queue params
      params["num_gpus"] = len(params["gpu_ids"])
      params["patch_size_x"] = params["patch_size_y"]
      params["effective_batch_size"] = params["num_gpus"]*params["batch_size"]
      params["batches_per_epoch"] = int(np.floor(params["epoch_size"]/params["effective_batch_size"]))
      params["w_shapes"] = [vals for vals in zip(params["patch_size_y"], params["patch_size_x"],
        params["input_channels"], params["output_channels"])]
      params["memristor_PCM_data_loc"] = "CAEs/data/Partial_Reset_PCM.pkl"

      #decoding is inverse of encoding
      params["w_shapes"] += params["w_shapes"][::-1]
      params["input_channels"] += params["input_channels"][::-1]
      params["output_channels"] += params["output_channels"][::-1]
      params["strides"] += params["strides"][::-1]
      params["num_layers"] = len(params["w_shapes"])
      params["min_after_dequeue"] = 0
      if not hasattr(params, "img_shape_x"):
        params["img_shape_x"] = params["img_shape_y"]
      params["im_shape"] = [params["img_shape_y"], params["img_shape_x"], params["num_colors"]]
      params["num_read_threads"] = params["num_threads"]# * params["num_gpus"] #TODO:only running on cpu - so don't mult by num_gpus?
      params["capacity"] = params["min_after_dequeue"] + (params["num_read_threads"] + 1) * params["effective_batch_size"]
      return params

    """Make output directories"""
    def make_dirs(self):
      if not os.path.exists(self.params["output_location"]+"/checkpoints/"):
        os.makedirs(self.params["output_location"]+"/checkpoints/")
      if not os.path.exists(self.params["weight_save_filename"]):
        os.makedirs(self.params["weight_save_filename"])

    """
    Resize images while preserving the aspect ratio then crop them to desired shape
    Resize is better than crop because it gets rid of possible compression artifacts in training data
    """
    def resize_preserving_aspect_then_crop(self, image):
      shape = tf.shape(image)
      orig_height = tf.to_float(shape[0])
      orig_width = tf.to_float(shape[1])
      orig_channels = tf.to_int32(shape[2])
      scale = tf.cond(tf.greater(orig_height, orig_width),
        lambda: tf.divide(self.params["img_shape_y"], orig_width),
        lambda: tf.divide(self.params["img_shape_x"], orig_height))
      new_height = tf.to_int32(tf.multiply(orig_height, scale))
      new_width = tf.to_int32(tf.multiply(orig_width, scale))
      image = tf.image.resize_images(image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
      image = tf.image.resize_image_with_crop_or_pad(image, self.params["img_shape_y"],
        self.params["img_shape_x"])
      return image

    """Preprocess a single image"""
    def preprocess_image(self, image):
      if self.params["num_colors"] == 1:
        image = tf.image.rgb_to_grayscale(image)
      image = tf.to_float(image)
      if self.params["downsample_images"]:
        if self.params["downsample_method"] == "resize":
          image.set_shape([None, None, self.params["num_colors"]])
          image = self.resize_preserving_aspect_then_crop(image)
        elif self.params["downsample_method"] == "crop":
          image = tf.image.resize_image_with_crop_or_pad(image, self.params["img_shape_y"], self.params["img_shape_x"])
        else:
          assert False, ("Parameter 'downsample_method' must be 'resize' or 'crop'")
      image = tf.div(image, 255.0)
      image = tf.subtract(image, tf.reduce_mean(image))
      return image

    """Function to load in a single image"""
    def read_image(self, filename_queue):
      # Read an entire image file at once
      image_reader = tf.WholeFileReader()
      filename, image_file = image_reader.read(filename_queue)
      # If the image has 1 channel (grayscale) it will broadcast to 3
      image = tf.image.decode_image(image_file, channels=3)
      preprocessed_image = self.preprocess_image(image)
      return [filename, preprocessed_image]

    def memristorize(self, u_in, memristor_std_eps):
      with tf.variable_scope("memristor_transform") as scope:
        path = self.params["memristor_PCM_data_loc"]
        #n_mem = tf.reduce_prod(u_out.get_shape()[1:])
        #n_mem = 32768 # 49152 for color, 32768 for grayscale
        if self.params["gauss_chan"] == True:
          get_channel_data = get_data.get_simulated_data
        else:
          get_channel_data = get_data.get_memristor_data
        (vs_data, mus_data, sigs_data,
          orig_VMIN, orig_VMAX, orig_RMIN,
          orig_RMAX) = get_channel_data(path, self.params["n_mem"], num_ext=5,
          norm_min=self.params["mem_v_min"], norm_max=self.params["mem_v_max"])
        v_clip = tf.clip_by_value(u_in, clip_value_min=self.params["mem_v_min"],
          clip_value_max=self.params["mem_v_max"])
        #v_trans = tensor_scaler(v_clip,orig_VMIN,orig_VMAX)
        r = mem_utils.memristor_output(v_clip, memristor_std_eps, vs_data, mus_data, sigs_data,
          interp_width=np.array(vs_data[1, 0] - vs_data[0, 0]).astype('float32'))
        #r_trans =  tensor_scaler(r, orig_RMIN, orig_RMAX)
        return tf.reshape(r, shape=u_in.get_shape(), name="mem_r")

    """Devisive normalizeation nonlinearity"""
    def gdn(self, layer_num, u_in, inverse):
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
        w_threshold = tf.where(tf.less(w_gdn, tf.constant(1e-3, dtype=tf.float32)), w_gdn_init,
          w_gdn)
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
    Estimate probability density, px, given a batch of latent vectors.
    """
   # def density_estimator(u_vals):
   #   with tf.variable_scope("density_weights"):
   #     psi = tf.get_variable(name="density_weights", shape=
   #   return (px, psi)

    """
    Make layer that does activation(conv(u,w)+b)
      where activation() is either relu or GDN
    """
    def layer_maker(self, layer_num, u_in, w_shape, w_init, stride, decode, relu, god_damn_network):
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
          if self.params["god_damn_network"]:
            u_in, b_gdn, w_gdn = self.gdn(layer_num, u_in, decode)
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
          if self.params["relu"]:
            u_out = tf.nn.relu(u_out, name="relu_activation")
        else:
          u_out = tf.add(tf.nn.conv2d(u_in, w, [1, stride, stride, 1],
            padding="SAME", use_cudnn_on_gpu=True), b, name="activation"+str(layer_num))
          if self.params["relu"]:
            u_out = tf.nn.relu(u_out, name="relu_activation")
          if self.params["god_damn_network"]:
            u_out, b_gdn, w_gdn = self.gdn(layer_num, u_out, decode)
      return w, b, u_out, b_gdn, w_gdn

    """
    Average gradients computed per GPU
    Args:
     grad_list: list of lists of (gradint, variable) tuples.
       Outer list is per tower, inner list is per varaible
    """
    def average_gradients(self, grad_list):
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
    def u_print(self, u_list):
      u_print_str = ""
      for idx, u in enumerate(u_list):
       u_eval = sess.run(u)
       u_shape = u_eval.shape
       num_u = u_eval.size
       u_print_str+="\tu"+str(idx)+"_shape: "+str(u_shape)+"\t"+str(num_u)+"\n"
       print(u_print_str)

    def construct_graph(self):
      self.graph = tf.Graph()
      with self.graph.as_default(),tf.device('/gpu:0'):
        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("optimizers") as scope:
          learning_rates = tf.train.exponential_decay(
            learning_rate=self.params["init_learning_rate"],
            global_step=self.global_step,
            decay_steps=self.params["decay_steps"],
            decay_rate=self.params["decay_rate"],
            staircase=self.params["staircase"],
            name="annealing_schedule")
          optimizer = tf.train.AdamOptimizer(learning_rates, name="grad_optimizer")

        with tf.variable_scope("placeholders") as scope:
          #n_mem = tf.reduce_prod(u_in.get_shape()[1:])
          #n_mem = 32768 # 49152 for color, 32768 for grayscale
          self.memristor_std_eps = tf.placeholder(tf.float32,
            shape=(self.params["effective_batch_size"], self.params["n_mem"]))
          self.triangle_centers = tf.placeholder(tf.float32, shape=[self.params["num_triangles"]], name="triangle_centers")
          self.quantization_noise = tf.placeholder(tf.float32, shape=(self.params["effective_batch_size"],
            self.params["n_mem"]), name="quantization_noise")

        with tf.variable_scope("queue") as scope:
          # Make a list of filenames. Strip removes "\n" at the end
          # file_location contains image locations separated by newlines
          filenames = tf.constant([string.strip()
            for string in open(self.params["file_location"], "r").readlines()])
          # Turn list of filenames into a string producer to feed names for each thread
          # Shuffling happens here - should be faster than shuffling after the images are loaded
          # Capacity is the max capacity of the queue - can be adjusted as needed
          fi_queue = tf.train.string_input_producer(filenames,
            shuffle=self.params["shuffle_inputs"], seed=self.params["seed"],
            capacity=self.params["capacity"])
          # We want to duplicate our filename_queue for each thread, to enforce thread safety
          fi_queue_threads  = [self.read_image(fi_queue)
            for _ in range(self.params["num_read_threads"])]
          # Batch join queues up images and delivers them a batch at a time
          filename_batch, self.x  = tf.train.batch_join(fi_queue_threads,
            batch_size=self.params["effective_batch_size"], capacity=self.params["capacity"], shapes=[[],
            [self.params["img_shape_y"], self.params["img_shape_x"], self.params["num_colors"]]])

        gradient_list = []
        # tf.get_variable_scope().reuse_variables() call will only apply
        # to items within this with statement
        with tf.variable_scope(tf.get_variable_scope()):
          for gpu_id in self.params["gpu_ids"]:
            with tf.device("/gpu:"+gpu_id):
              with tf.name_scope("tower_"+gpu_id) as scope:
                self.w_list = []
                self.u_list = [self.x]
                self.b_list = []
                self.b_gdn_list = []
                self.w_gdn_list = []
                self.mle_thetas = ef.thetas(self.params["n_mem"], self.params["num_triangles"])
                self.reset_mle_thetas = self.mle_thetas.assign(tf.ones((self.params["n_mem"], self.params["num_triangles"])))
                w_inits = [tf.contrib.layers.xavier_initializer_conv2d(uniform=False,
                  seed=self.params["seed"], dtype=tf.float32)
                  for _ in np.arange(self.params["num_layers"]/2)]
                w_inits += w_inits # decode inits are the same as encode inits
                w_shapes_strides_list = zip(self.params["w_shapes"], self.params["strides"])
                for layer_idx, w_shapes_strides in enumerate(w_shapes_strides_list):
                  decode = False if layer_idx < self.params["num_layers"]/2 else True
                  w, b, u_out, b_gdn, w_gdn = self.layer_maker(layer_idx, self.u_list[layer_idx],
                    w_shapes_strides[0], w_inits[layer_idx], w_shapes_strides[1], decode,
                    self.params["relu"], self.params["god_damn_network"])
                  if layer_idx == self.params["num_layers"]/2-1:
                    u_resh = tf.reshape(u_out, [self.params["effective_batch_size"], self.params["n_mem"]])
                    ll = ef.log_likelihood(u_resh, self.mle_thetas, self.triangle_centers)
                    self.mle_update = ef.mle(ll, self.mle_thetas, self.params["mle_lr"])
                    self.u_probs = ef.prob_est(u_resh, self.mle_thetas, self.triangle_centers)
                    self.latent_entropies = ef.calc_entropy(self.u_probs)
                    with tf.variable_scope("loss") as scope:
                      self.reg_loss = tf.reduce_mean(tf.reduce_sum(self.params["GAMMA"]
                        * (tf.nn.relu(u_out - self.params["mem_v_max"])
                        + tf.nn.relu(self.params["mem_v_min"] - u_out)), axis=[1,2,3]))
                    if self.params["memristorify"]:
                      gpu_index = int(gpu_id) if len(self.params["gpu_ids"]) > 1 else 0
                      memristor_std_eps_slice = tf.split(value=self.memristor_std_eps,
                        num_or_size_splits=self.params["num_gpus"], axis=0)[gpu_index]
                      u_out = self.memristorize(u_out, memristor_std_eps_slice)
                    else: # add uniform noise if not using channels
                      u_out = tf.reshape(tf.add(u_resh, self.quantization_noise, name="noisy_latent_u"),
                        u_out.get_shape())
                  self.w_list.append(w)
                  self.u_list.append(u_out)
                  self.b_list.append(b)
                  self.b_gdn_list.append(b_gdn)
                  self.w_gdn_list.append(w_gdn)

                with tf.variable_scope("loss") as scope:
                  self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(self.u_list[0],
                    self.u_list[-1])), axis=[1,2,3]))
                  self.ent_loss = self.params["LAMBDA"] * tf.reduce_sum(self.latent_entropies)
                  loss_list = [self.recon_loss, self.reg_loss, self.ent_loss]
                  self.total_loss = tf.add_n(loss_list, name="total_loss")

                with tf.variable_scope("optimizers") as scope:
                  self.train_vars = self.w_list + self.b_list
                  self.train_vars += self.b_gdn_list + self.w_gdn_list if self.params["god_damn_network"] else []
                  grads_and_vars = optimizer.compute_gradients(self.total_loss,
                    var_list=self.train_vars)
                  gradient_list.append(grads_and_vars)

                tf.get_variable_scope().reuse_variables()

        avg_grads = self.average_gradients(gradient_list)
        with tf.variable_scope("optimizers") as scope:
          self.train_op = optimizer.apply_gradients(avg_grads, global_step=self.global_step)

        with tf.name_scope("savers") as scope:
          self.full_saver = tf.train.Saver(var_list=self.train_vars, max_to_keep=self.params["num_epochs"])

        with tf.name_scope("performance_metrics") as scope:
          self.MSE = tf.reduce_mean(tf.square(tf.subtract(tf.multiply(self.u_list[0], 255.0),
            tf.multiply(tf.clip_by_value(self.u_list[-1], clip_value_min=-1.0, clip_value_max=1.0), 255.0))),
            reduction_indices=[1,2,3], name="mean_squared_error")
          self.batch_MSE = tf.reduce_mean(self.MSE, name="batch_mean_squared_error")
          self.SNRdB = tf.multiply(10.0, tf.log(tf.div(tf.square(tf.nn.moments(self.u_list[0],
            axes=[0,1,2,3])[1]), self.batch_MSE)), name="recon_quality")

        with tf.name_scope("summaries") as scope:
          tf.summary.image("input", self.u_list[0])
          tf.summary.image("reconstruction",self.u_list[-1])
          [tf.summary.histogram("u"+str(idx),u) for idx,u in enumerate(self.u_list)]
          [tf.summary.histogram("w"+str(idx),w) for idx,w in enumerate(self.w_list)]
          [tf.summary.histogram("b"+str(idx),b) for idx,b in enumerate(self.b_list)]
          if self.params["god_damn_network"]:
            [tf.summary.histogram("b_gdn"+str(idx),u) for idx,u in enumerate(self.b_gdn_list)]
            [tf.summary.histogram("w_gdn"+str(idx),w) for idx,w in enumerate(self.w_gdn_list)]
          tf.summary.scalar("total_loss", self.total_loss)
          tf.summary.scalar("batch_MSE", self.batch_MSE)
          tf.summary.scalar("SNRdB", self.SNRdB)

        # Must initialize local variables as well as global to init num_epochs
        # in tf.train.string_input_produce
        self.merged_summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.params["output_location"], self.graph)
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
