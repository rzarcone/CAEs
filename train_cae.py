import matplotlib
matplotlib.use("Agg")
import os
import tensorflow as tf
import utils.plot_functions as pf
import numpy as np
from cae_model import cae

params = {}
#shitty hard coding
params["n_mem"] = 7680  #32768 #49152 for color, 32768 for grayscale

#general params
params["run_name"] = "test_model"
#params["file_location"] = "/media/tbell/datasets/natural_images.txt"
params["file_location"] = "/media/tbell/datasets/test_images.txt"
params["gpu_ids"] = ["0"]
params["output_location"] = os.path.expanduser("~")+"/CAE_Project/CAEs/model_outputs/"+params["run_name"]
params["num_threads"] = 6
params["num_epochs"] = 20
#params["epoch_size"] = 112682
params["epoch_size"] = 49900
params["eval_interval"] = 100
params["seed"] = 1234567890

#checkpoint params
params["run_from_check"] = True
params["check_load_run_name"] = "7680_med_compress_pcm"
params["check_load_path"] = os.path.expanduser("~")+"/CAE_Project/CAEs/model_outputs/"+params["check_load_run_name"]+"/checkpoints/chkpt_ep39-45040"

#image params
params["shuffle_inputs"] = True
params["batch_size"] = 100
params["img_shape_y"] = 256
params["num_colors"] = 1
params["downsample_images"] = True
params["downsample_method"] = "resize" # can be "crop" or "resize"

#learning rates
params["init_learning_rate"] = 5.0e-4
params["decay_steps"] = 10000#epoch_size*0.5*num_epochs #0.5*epoch_size
params["staircase"] = True
params["decay_rate"] = 0.9

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
params["gauss_chan"] = False

cae_model = cae(params)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False # for debugging - log devices used by each variable

with tf.Session(config=config, graph=cae_model.graph) as sess:
  sess.run(cae_model.init_op)
  if cae_model.params["run_from_check"] == True:
    cae_model.full_saver.restore(sess, cae_model.params["check_load_path"])
  # Coordinator manages threads, checks for stopping requests
  coord = tf.train.Coordinator()
  # queue_runners are created by helper functions tf.train.string_input_producer() and tf.train.batch_join()
  enqueue_threads = tf.train.start_queue_runners(sess, coord=coord, start=True)
  for epoch_idx in range(cae_model.params["num_epochs"]):
    for i in range(cae_model.params["batches_per_epoch"]):
      #n_mem_eval = sess.run(tf.to_int32(tf.size(self.u_list[int(num_layers/2)])/self.u_list[int(num_layers/2)].get_shape()[0]))
      #n_mem_eval =49152 # 49152
      mem_std_eps = np.random.standard_normal((cae_model.params["effective_batch_size"],
        cae_model.params["n_mem"])).astype(np.float32)
      feed_dict={cae_model.memristor_std_eps:mem_std_eps}
      _, step = sess.run([cae_model.train_op, cae_model.global_step], feed_dict=feed_dict)
      if step % cae_model.params["eval_interval"] == 0:
        #loss_list = [cae_model.recon_loss, cae_model.reg_loss, cae_model.total_loss]
        #model_vars = loss_list + [cae_model.merged_summaries, cae_model.batch_MSE]
        #output_list = sess.run(model_vars, feed_dict=feed_dict)
        #cae_model.train_writer.add_summary(output_list[3], step)
        #print("step %04d\treg_loss %03g\trecon_loss %g\ttotal_loss %g\tMSE %g"%(
        #  step, output_list[1], output_list[0], output_list[2], output_list[4]))
        model_vars = [cae_model.merged_summaries, cae_model.reg_loss, cae_model.recon_loss,
          cae_model.total_loss, cae_model.batch_MSE]
        [summary, ev_reg_loss, ev_recon_loss, ev_total_loss, mse] = sess.run(model_vars, feed_dict)
        cae_model.train_writer.add_summary(summary, step)
        print("step %04d\treg_loss %03g\trecon_loss %g\ttotal_loss %g\tMSE %g"%(
          step, ev_reg_loss, ev_recon_loss, ev_total_loss, mse))
        #u_print(self.u_list)
    cae_model.full_saver.save(sess, save_path=cae_model.params["output_location"]+"/checkpoints/chkpt",
      global_step=cae_model.global_step)
    w_enc_eval = np.squeeze(sess.run(tf.transpose(cae_model.w_list[0], perm=[3,0,1,2])))
    pf.save_data_tiled(w_enc_eval, normalize=True, title="Weights0",
      save_filename=cae_model.params["weight_save_filename"]+"/Weights_enc_ep"+str(epoch_idx)+".png")
    w_dec_eval = np.squeeze(sess.run(tf.transpose(cae_model.w_list[-1], perm=[3,0,1,2])))
    pf.save_data_tiled(w_dec_eval, normalize=True, title="Weights-1",
      save_filename=cae_model.params["weight_save_filename"]+"/Weights_dec_ep"+str(epoch_idx)+".png")
  coord.request_stop()
  coord.join(enqueue_threads)
