import numpy as np
import matplotlib.pyplot as plt
import csv

data_path = '/home/rzarcone/CAE_Project/CAEs/data/R_D_for_Simoncelli.csv'
fig_path = 'test_r_d_curve.png'
R_D_list = []

with open(data_path, 'rt') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        R_D_list.append(row)

R_D_array = np.array(R_D_list)
R_D_new = R_D_array[:,2:4]

fig = plt.figure()
plt.scatter(R_D_new[:,0],R_D_new[:,1])
fig.savefig(fig_path)
plt.close(fig)
