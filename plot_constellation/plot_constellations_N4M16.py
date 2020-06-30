import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

data = sio.loadmat("./cons_data/cons_N4M16.mat")
U4 = data['U']

data = sio.loadmat("./cons_data/cons_S_OFDM_N4M2.mat")
U2 = data['U']

data = sio.loadmat("./cons_data/cons_S_OFDM_IM_N4K1M4.mat")
U3 = data['U']

data = sio.loadmat("./cons_data/cons_OFDM_IM_N4K1M4.mat")
U1 = data['U']

marker_size = 16

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8, 8))

ax1.plot(U1[:,0,0], U1[:,0,1], "b.", markersize=marker_size)
ax1.plot(U1[:,1,0], U1[:,1,1], "rx", markersize=marker_size)
ax1.grid(True)
ax1.set_title('OFDM-IM')


ax2.plot(U2[:,0,0], U2[:,0,1], "b.", markersize=marker_size)
ax2.plot(U2[:,1,0], U2[:,1,1], "rx", markersize=marker_size)
ax2.grid(True)
ax2.set_title('S-OFDM')


ax3.plot(U3[:,0,0], U3[:,0,1], "b.", markersize=marker_size)
ax3.plot(U3[:,1,0], U3[:,1,1], "rx", markersize=marker_size)
ax3.grid(True)
ax3.set_title('S-OFDM-IM', fontsize=12)

ax4.plot(U4[:,0,0], U4[:,0,1], "b.", markersize=marker_size)
ax4.plot(U4[:,1,0], U4[:,1,1], "rx", markersize=marker_size)
major_ticks = np.arange(-1, 2, 1)    
ax4.set_xticks(major_ticks)
ax4.set_yticks(major_ticks)
ax4.grid(True)
ax4.set_title('Proposed MC-AE', fontsize=12)


# fig.suptitle('Contellation for N=2 M=4', fontsize=12)
fig.savefig('./cons_figs/cons_N4M16.png',dpi=100)
plt.show()