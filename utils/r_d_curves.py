import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# csv format is image_number, bits_per_pixel, mem_per_pixel, PSNR, MS-SSIM, MSE
data_folder = os.path.expanduser("~")+"/CAE_Project/CAEs/data/"

balle_file = data_folder+"r_d_Balle.csv"
jpeg_file = data_folder+"r_d_JPEG.csv"
jpeg2k_file = data_folder+"r_d_JPEG2k.csv"

proposed_file0 = data_folder+"r_d_proposed_3072_max_compress_pcm.csv"
proposed_file1 = data_folder+"r_d_proposed_7680_med_compress_pcm.csv"
proposed_file2 = data_folder+"r_d_proposed_32768_min_compress_pcm.csv"

proposed_file3 = data_folder+"r_d_proposed_3072_max_compress_gauss.csv"
proposed_file4 = data_folder+"r_d_proposed_7680_med_compress_gauss.csv"
proposed_file5 = data_folder+"r_d_proposed_32768_min_compress_gauss.csv"

proposed_file6 = data_folder+"r_d_proposed_3072_max_compress_pcm_relu.csv"
proposed_file7 = data_folder+"r_d_proposed_7680_med_compress_pcm_relu.csv"

fig_path = data_folder+"proposed_pcm_balle_r_d_curve.pdf"

jpeg2k_r_d_list = []
with open(jpeg2k_file, 'rt') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  for row_idx, row in enumerate(reader):
    if row_idx > 0: # first row is header
      jpeg2k_r_d_list.append([float(val) for val in row])
jpeg2k_array = np.array(jpeg2k_r_d_list)

balle_r_d_list = []
with open(balle_file, 'rt') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  for row_idx, row in enumerate(reader):
    if row_idx > 0: # first row is header
      balle_r_d_list.append([float(val) for val in row])
balle_array = np.array(balle_r_d_list)

proposed_r_d_list = []
with open(proposed_file0, 'rt') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  for row_idx, row in enumerate(reader):
    if row_idx > 0: # first row is header
      proposed_r_d_list.append([float(val) if val != "NA" else 0 for val in row])
proposed_array0 = np.array(proposed_r_d_list)

proposed_r_d_list = []
with open(proposed_file1, 'rt') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  for row_idx, row in enumerate(reader):
    if row_idx > 0: # first row is header
      proposed_r_d_list.append([float(val) if val != "NA" else 0 for val in row])
proposed_array1 = np.array(proposed_r_d_list)

proposed_r_d_list = []
with open(proposed_file2, 'rt') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  for row_idx, row in enumerate(reader):
    if row_idx > 0: # first row is header
      proposed_r_d_list.append([float(val) if val != "NA" else 0 for val in row])
proposed_array2 = np.array(proposed_r_d_list)

fig = plt.figure()
plt.scatter(jpeg2k_array[:,2], jpeg2k_array[:,5], s=18, c="g", edgecolors="none", alpha=0.25)
plt.scatter(balle_array[:,2], balle_array[:,5], s=32, c="b", edgecolors="none", alpha=0.15)
plt.scatter(proposed_array0[:,2], proposed_array0[:,5], s=19, c="r", edgecolors="none", alpha=0.25)
plt.scatter(proposed_array1[:,2], proposed_array1[:,5], s=19, c="k", edgecolors="none", alpha=0.25)
plt.scatter(proposed_array2[:,2], proposed_array2[:,5], s=19, c="c", edgecolors="none", alpha=0.25)
plt.ylabel("MSE")
plt.xlabel("Memristors Per Pixel")
plt.ylim([0, 450])
plt.xlim([0, 1.0])
plt.legend(["JPEG2k", "Balle", "Proposed_dn_pcm_max", "Proposed_dn_pcm_med", "Proposed_dn_pcm_min"])
#plt.legend(["JPEG2k", "Balle", "Proposed"])
#plt.legend(["JPEG2k", "Balle", "Proposed"])
fig.savefig(fig_path)
plt.close(fig)
