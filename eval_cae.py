import matplotlib
matplotlib.use("Agg")

import os
import csv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils.plot_functions as pf
from cae_model import cae

params = {}
#shitty hard coding
params["n_mem"] = 7680  #32768 #49152 for color, 32768 for grayscale

#general params
params["run_name"] = "eval_train_boot_from_22800"
params["file_location"] = "/media/tbell/datasets/kodak/image_list.txt"
params["gpu_ids"] = ["0"]
params["output_location"] = os.path.expanduser("~")+"/CAE_Project/CAEs/"+params["run_name"]
params["weight_save_filename"] = params["output_location"]+"/weights/"
params["num_threads"] = 6
params["num_epochs"] = 1
params["epoch_size"] = 24
params["eval_interval"] = 1
params["seed"] = 1234567890
params["check_load_path"] = "/home/dpaiton/CAE_Project/CAEs/train/checkpoints/chkpt_-22800"
params["run_from_check"] = True

#image params
params["shuffle_inputs"] = True
params["batch_size"]= 24
params["img_shape_y"] = 256
params["num_colors"] = 1

#learning rates
params["init_learning_rate"] = 0.0
params["decay_steps"] = 0.0
params["staircase"] = True
params["decay_rate"] = 1.0

#layer params
params["memristorify"] = True
params["god_damn_network"] = True
params["relu"] = False

#layer dimensions
params["input_channels"] = [params["num_colors"], 128, 128]
params["output_channels"] = [128, 128, 30]
params["patch_size_y"] = [9, 5, 5]
params["strides"] = [4, 2, 2]

#memristor params
params["GAMMA"] = 1.0  # slope of the out of bounds cost
params["mem_v_min"] = -1.0
params["mem_v_max"] = 1.0

cae_model = cae(params)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False # for debugging - log devices used by each variable

with tf.Session(config=config, graph=cae_model.graph) as sess:
  sess.run(cae_model.init_op)
  cae_model.full_saver.restore(sess, cae_model.params["check_load_path"])
  # Load files
  coord = tf.train.Coordinator()
  enqueue_threads = tf.train.start_queue_runners(sess, coord=coord, start=True)
  # set memristor noise
  mem_std_eps = np.random.standard_normal((cae_model.params["effective_batch_size"],
    cae_model.params["n_mem"])).astype(np.float32)
  feed_dict={cae_model.memristor_std_eps:mem_std_eps}
  tf_var_list = cae_model.train_vars + cae_model.u_list
  eval_list = tf_var_list + [cae_model.total_loss, cae_model.recon_loss,
    cae_model.reg_loss, cae_model.MSE, cae_model.SNRdB]
  out_vars = sess.run(eval_list, feed_dict=feed_dict)

  num_img_pixels = cae_model.params["img_shape_y"]*cae_model.params["img_shape_x"]*cae_model.params["num_colors"]
  rate = cae_model.params["n_mem"]/num_img_pixels
  mse = out_vars[-2]
  csv_file_loc = os.path.expanduser("~")+"/CAE_Project/CAEs/data/R_D_for_Proposed.csv"
  with open(csv_file_loc, "w") as csv_file:
    writer = csv.writer(csv_file, dialect="excel", delimiter=",")
    for mse_val in mse:
      writer.writerow([rate, mse_val])

  coord.request_stop()
  coord.join(enqueue_threads)
