import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# csv format is rate, PSNR, rate, MSE
data_folder = os.path.expanduser("~")+"/CAE_Project/CAEs/data/"
simoncelli_file = data_folder+"R_D_for_Simoncelli.csv"
#toderici_file = data_folder+"toderici_bitrates.csv" # https://github.com/tensorflow/models/tree/master/compression/image_encoder
proposed_file = data_folder+"R_D_for_Proposed.csv"
fig_path = data_folder+"test_r_d_curve.png"

#toderici_R_D_list = []
#with open(toderici_file, 'rt') as csvfile:
#    spamreader = csv.reader(csvfile, delimiter=',')
#    for row in spamreader:
#        toderici_R_D_list.append(row)
#toderici_array = np.array(toderici_R_D_list)

simoncelli_R_D_list = []
with open(simoncelli_file, 'rt') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        simoncelli_R_D_list.append(row)
simoncelli_array = np.array(simoncelli_R_D_list)[:,2:] # only want MSE

proposed_R_D_list = []
with open(proposed_file, 'rt') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        proposed_R_D_list.append(row)
proposed_array = np.array(proposed_R_D_list)

fig = plt.figure()
plt.scatter(simoncelli_array[:,0], simoncelli_array[:,1], c="b")
plt.scatter(proposed_array[:,0], proposed_array[:,1], c="r")
fig.savefig(fig_path)
plt.close(fig)
