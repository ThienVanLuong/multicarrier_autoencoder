import scipy.io as sio
import numpy as np

data = sio.loadmat("./cons_data/cons_N2M4_awgn.mat")
# data = sio.loadmat("./cons_data/Cons_S_OFDM_IM_N2K1M4.mat")

U = data['U']
V = U[:,:,0] + 1j*U[:,:,1]
M = V.shape[0]
print(np.round(V,2))

dis = []
for i in range(M):
    for j in range(M):
        if j>i:
            dis_ij = np.sqrt(np.sum(np.abs(V[j]-V[i])*np.abs(V[j]-V[i])))
            dis.append(dis_ij)

print('\nMinimum distance: ', min(dis))
print('\nAll distances: ', np.round(dis,2))
