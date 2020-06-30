import scipy.io as sio
import numpy as np


# data = sio.loadmat("./cons_data/cons_S_OFDM_IM_N2K1M4.mat")
data = sio.loadmat("./cons_data/cons_N4M16.mat")

U = data['U']
V = U[:,:,0] + 1j*U[:,:,1]
print('Codewords: \n', np.round(V,4))

M = len(V)
N = V.shape[1]
Gd = N

Gc = 10000
sum_Gc = 0
for n1 in range(M):
    for n2 in range(n1+1,M):
        Y = np.prod(np.power(np.abs(V[n1]-V[n2]),2))
        sum_Gc = sum_Gc + 1/Y
        if (Y<Gc):
            Gc = Y           
Gcc = np.power(sum_Gc/M/(2**(2*Gd)),-1/N)
print('\nCoding gain = ', Gcc)