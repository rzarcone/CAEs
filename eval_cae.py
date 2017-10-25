import matplotlib
matplotlib.use("Agg")

import os
import csv
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from cae_model import cae
import DeepSparseCoding.utils.image_processing as ip
import DeepSparseCoding.utils.plot_functions as pf

params = {}
#shitty hard coding
params["n_mem"] = 32768#32768#3072 for max, 7680 for med, 32768 for min

#checkpoints

#params["check_load_run_name"] = "3072_max_compress_pcm" #ep49_56300
#params["check_load_run_name"] = "7680_med_compress_pcm" #ep39_45040
params["check_load_run_name"] = "32768_min_compress_pcm" #ep49-56300

#params["check_load_run_name"] = "3072_max_compress_gauss" #ep49_56300
#params["check_load_run_name"] = "7680_med_compress_gauss" #ep49-56300
#params["check_load_run_name"] = "32768_min_compress_gauss" #ep49-56300

#params["check_load_run_name"] = "3072_max_compress_pcm_relu" #ep49_56300 # GDN3 Not Found
#params["check_load_run_name"] = "7680_med_compress_pcm_relu" #ep49-56300

#general params
params["run_name"] = "eval_"+params["check_load_run_name"]
#params["file_location"] = "/media/tbell/datasets/kodak/image_list.txt"
params["file_location"] = "/media/tbell/datasets/test_images.txt"
params["gpu_ids"] = ["0"]
params["output_location"] = os.path.expanduser("~")+"/CAE_Project/CAEs/model_outputs/"+params["run_name"]
params["num_threads"] = 1
params["num_epochs"] = 1
#params["epoch_size"] = 24
params["epoch_size"] = 49900
params["eval_interval"] = 1
params["seed"] = 1234567890
params["check_load_path"] = "/home/dpaiton/CAE_Project/CAEs/model_outputs/"+params["check_load_run_name"]+"/checkpoints/chkpt_ep49-56300"
params["run_from_check"] = True

#image params
params["shuffle_inputs"] = False
params["batch_size"]= 24
params["img_shape_y"] = 256
#params["img_shape_x"] = 256
params["num_colors"] = 1
params["downsample_images"] = True
params["downsample_method"] = "crop" # can be "crop" or "resize"

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
params["output_channels"] = [128, 128, 128] #12 for max, 30 for med, 128 for min
params["patch_size_y"] = [9, 5, 5]
params["strides"] = [4, 2, 2]

#memristor params
params["GAMMA"] = 1.0  # slope of the out of bounds cost
params["mem_v_min"] = -1.0
params["mem_v_max"] = 1.0
params["gauss_chan"] = True

cae_model = cae(params)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False # for debugging - log devices used by each variable

#TODO: set up for-loop over batches to agg correlation matrix, save corr matrix, load in ipynb
with tf.Session(config=config, graph=cae_model.graph) as sess:
  sess.run(cae_model.init_op)
  cae_model.full_saver.restore(sess, cae_model.params["check_load_path"])
  # Load files
  coord = tf.train.Coordinator()
  enqueue_threads = tf.train.start_queue_runners(sess, coord=coord, start=True)
  # set memristor noise
  mem_std_eps = np.random.standard_normal((cae_model.params["effective_batch_size"],
    cae_model.params["n_mem"])).astype(np.float32)
  feed_dict = {cae_model.memristor_std_eps:mem_std_eps}
  tf_var_list = cae_model.train_vars + cae_model.u_list
  loss_list = [cae_model.total_loss, cae_model.reg_loss, cae_model.recon_loss]
  eval_list = tf_var_list + loss_list + [cae_model.MSE, cae_model.SNRdB]
  out_vars = sess.run(eval_list, feed_dict=feed_dict)

  ulist = sess.run(cae_model.u_list, feed_dict)
  pickle.dump(ulist, open("/home/dpaiton/CAE_Project/"+params["check_load_run_name"]+"_ulist.npz", "wb"))
  #np.savez("/home/dpaiton/CAE_Project/"+params["check_load_run_name"]+"_ulist.npz", data=ulist)

  weights = sess.run(cae_model.train_vars, feed_dict)
  np.savez("/home/dpaiton/CAE_Project/"+params["check_load_run_name"]+"_weights.npz", data=weights)

  import IPython; IPython.embed(); raise SystemExit

  num_img_pixels = cae_model.params["img_shape_y"]*cae_model.params["img_shape_x"]*cae_model.params["num_colors"]
  mem_per_pixel = cae_model.params["n_mem"]/num_img_pixels
  mse = out_vars[-2]
  csv_file_loc = os.path.expanduser("~")+"/CAE_Project/CAEs/data/r_d_proposed_"+params["check_load_run_name"]+".csv"
  with open(csv_file_loc, "w") as csv_file:
    writer = csv.writer(csv_file, dialect="excel", delimiter=",")
    writer.writerow(["image_number","bits_per_pixel","mem_per_pixel","PSNR","MS-SSIM","MSE"])
    for img_num, mse_val in enumerate(mse):
      writer.writerow([str(img_num).zfill(2), "NA", mem_per_pixel, "NA", "NA", mse_val])

  coord.request_stop()
  coord.join(enqueue_threads)
