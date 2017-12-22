import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.python import debug as tf_debug
from cae_model import cae

params = {}
#shitty hard coding
params["n_mem"] = 7680  #32768 #49152 for color, 32768 for grayscale

#general params
params["run_name"] = "debug_ent_test_med_compress"
#params["file_location"] = "/media/tbell/datasets/natural_images.txt"
params["file_location"] = "/media/tbell/datasets/test_images.txt"
params["gpu_ids"] = ["0"]#['0','1']
params["output_location"] = os.path.expanduser("~")+"/CAE_Project/CAEs/model_outputs/"+params["run_name"]
params["num_threads"] = 6
params["num_epochs"] = 20
#params["epoch_size"] = 112682
params["epoch_size"] = 49900
params["eval_interval"] = 1
params["seed"] = 1234567890

#checkpoint params
params["run_from_check"] = False 
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
params["memristorify"] = False
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

#entropy params
params["LAMBDA"] = 0.1
params["num_triangles"] = 20
params["mle_lr"] = 0.1
params["num_mle_steps"] = 5 
params["quant_noise_scale"] = 1.0/128.0 # simulating quantizing u in {-1.0, 1.0} to uint8 (256 values)
mle_triangle_centers = np.linspace(params["mem_v_min"], params["mem_v_max"], params["num_triangles"])

cae_model = cae(params)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False # for debugging - log devices used by each variable


with tf.Session(config=config, graph=cae_model.graph) as sess:
  sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

  sess.run(cae_model.init_op)
  if cae_model.params["run_from_check"] == True:
    cae_model.full_saver.restore(sess, cae_model.params["check_load_path"])
  # Coordinator manages threads, checks for stopping requests
  coord = tf.train.Coordinator()
  # queue_runners are created by helper functions tf.train.string_input_producer() and tf.train.batch_join()
  enqueue_threads = tf.train.start_queue_runners(sess, coord=coord, start=True)
  feed_dict={cae_model.triangle_centers:mle_triangle_centers}
  for epoch_idx in range(cae_model.params["num_epochs"]):
    for batch_idx in range(cae_model.params["batches_per_epoch"]):
      # Add quantization noise if not using a fixed channel
      if not params["memristorify"] and not params["gauss_chan"]:
        quant_noise = np.random.uniform(-params["quant_noise_scale"], params["quant_noise_scale"],
          size=(cae_model.params["effective_batch_size"], cae_model.params["n_mem"]))
        feed_dict[cae_model.quantization_noise] = quant_noise
      else:
        mem_std_eps = np.random.standard_normal((cae_model.params["effective_batch_size"],
          cae_model.params["n_mem"])).astype(np.float32)
        feed_dict[cae_model.memristor_std_eps] = mem_std_eps
      # Update MLE estimate
      sess.run(cae_model.reset_mle_thetas, feed_dict)
      for mle_step in range(params["num_mle_steps"]):
        sess.run(cae_model.mle_update, feed_dict)
      # Update network weights
      _, step = sess.run([cae_model.train_op, cae_model.global_step], feed_dict)
      # Eval model
      if step % cae_model.params["eval_interval"] == 0:
        model_vars = [cae_model.merged_summaries, cae_model.reg_loss, cae_model.recon_loss,
          cae_model.ent_loss, cae_model.total_loss, cae_model.batch_MSE]
        [summary, ev_reg_loss, ev_recon_loss, ev_ent_loss, ev_total_loss, mse] = sess.run(model_vars, feed_dict)
        cae_model.train_writer.add_summary(summary, step)
        print("step %04d\treg_loss %03g\trecon_loss %g\tent_loss %g\ttotal_loss %g\tMSE %g"%(
          step, ev_reg_loss, ev_recon_loss, ev_ent_loss, ev_total_loss, mse))
        
    #Checkpoint and save image of weights each epoch
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
