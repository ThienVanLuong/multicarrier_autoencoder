import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


# data = sio.loadmat("./cons_data/cons_N2M4_awgn.mat")
data = sio.loadmat("./cons_data/cons_N2M4.mat")
U4 = data['U']

U3 = np.array([[1,-1, 0, 0], [0, 0, -1, 1]]) #S-OFDM-IM
U1 = np.array([[1,-1], [0, 0]]) #OFDM
U2 = np.sqrt(1)*np.array([[1,-1, 0], [0, 0, 0]]) #OFDM-IM

marker_size = 16

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8, 8))
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8, 8))


ax4.plot(U4[:,0,0], U4[:,0,1], "b.", markersize=marker_size)
ax4.plot(U4[:,1,0], U4[:,1,1], "rx", markersize=marker_size)
major_ticks = np.arange(-1, 2, 0.5)    
ax4.set_xticks(major_ticks)
ax4.set_yticks(major_ticks)
ax4.grid(True)
ax4.set_title('Proposed MC-AE', fontsize=12)

ax3.plot(U3[0,:], U3[1,:], "b.", markersize=marker_size)
ax3.plot(U3[0,:], U3[1,:], "rx", markersize=marker_size)
ax3.grid(True)
ax3.set_title('S-OFDM/S-OFDM-IM', fontsize=12)

ax1.plot(U1[0,:], U1[1,:], "b.", markersize=marker_size)
ax1.plot(U1[0,:], U1[1,:], "rx", markersize=marker_size)
ax1.grid(True)
ax1.set_title('OFDM')

ax2.plot(U2[0,:], U2[1,:], "b.", markersize=marker_size)
ax2.plot(U2[0,:], U2[1,:], "rx", markersize=marker_size)
ax2.grid(True)
ax2.set_title('OFDM-IM')

# fig.suptitle('Contellation for N=2 M=4', fontsize=12)
fig.savefig('./cons_figs/cons_N2M4xx.png',dpi=100)
plt.show()