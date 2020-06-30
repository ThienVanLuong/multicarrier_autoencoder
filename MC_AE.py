""" 
- This is the implementation of MC-AE under Rayleigh fading channel
- Requirements: tensorflow 1.15, keras 2.0.8
- Created by Thien Van Luong, Research Fellow at University of Southampton, UK. 
- Email: thien.luong@soton.ac.uk.
"""
from utils import generate_one_hot_vectors, calculate_coding_gain
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Lambda, Add, Reshape, Dense, Flatten, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K

# ----------------- MC-AE parameters setup ---------------------------
N = 4                   # number of subcarriers
M = 16                   # total modulation size
m = int(np.log2(M))     # bits per codeword/block of N subcarriers
R = m/N                 # bit rate - bits/s/Hz

# neural network config for training and testing
SNRdB_train = 7         # training SNR
b_size = 512            # batch_size
pre_trained = False     # use pre-trained model or not
norm_type = 0           # 0 per sample norm, 1 per batch norm
n_epoch = 100           # number of epochs
l_rate = 0.001          # learning rate
hidden_layer_dec = np.array([256])   # number of nodes of hidden layers in decoder
f_loss = 'mean_squared_error'       # categorical_crossentropy/ mean_squared_error loss function
train_size = 20000      # number of training data samples
test_size = 20000       # number of testing data samples
act_enc = 'linear'      # activation used in the encoder
act_dec = 'relu'        # activation used in the decoder
ini = 'normal'          # weight initialization method


snr_train = 10**(SNRdB_train/10.0)
noise_std = np.sqrt(1/(2*R*snr_train))

"""==== build channel layers for training and testing ====""" 
# training channel layer
def channel(Z):
    # eps_train = 1/(1+10**2) # trained with imperfect CSI
    eps_train = 0
    
    H_est = K.random_normal(K.shape(Z),mean=0,stddev=np.sqrt(1-eps_train))/np.sqrt(2)
    H_R_est = H_est[:,:,0]
    H_I_est = H_est[:,:,1]
    err = K.random_normal(K.shape(Z), mean=0, stddev=np.sqrt(eps_train))/np.sqrt(2)
    
    H = H_est + err
    
    H_R = H[:,:,0]
    H_I = H[:,:,1]
    real = H_R * Z[:,:,0] - H_I* Z[:,:,1]
    imag = H_R * Z[:,:,1] + H_I* Z[:,:,0]
    
    noise_r = K.random_normal(K.shape(real),mean=0,stddev=noise_std)                                                   
    noise_i = K.random_normal(K.shape(imag),mean=0,stddev=noise_std)
    
    real = Add()([real, noise_r])
    imag = Add()([imag, noise_i])   
    
    Y = K.stack([real, imag, H_R_est, H_I_est], axis=2)  
    return Y

# testing channel layer
def channel_test(Z, noise_std, test_size, imperfect_channel=False):
    eps = 0
    if imperfect_channel:
        snr = noise_std*noise_std*2*R
        eps = 1/(1+1/snr) # imperfect CSI
        
    H_R_est = np.random.normal(0, np.sqrt(1-eps), (test_size, N))/np.sqrt(2)
    H_I_est = np.random.normal(0, np.sqrt(1-eps), (test_size, N))/np.sqrt(2) 
    
    err_r = K.random_normal(K.shape(Z[:,:,0]), mean=0, stddev=np.sqrt(eps))/np.sqrt(2)
    err_i = K.random_normal(K.shape(Z[:,:,1]), mean=0, stddev=np.sqrt(eps))/np.sqrt(2)
    
    H_R = H_R_est + err_r
    H_I = H_I_est + err_i
    
    real = H_R*Z[:,:,0] - H_I*Z[:,:,1] # x.shape (50000, 2, 2)
    imag = H_R*Z[:,:,1] + H_I*Z[:,:,0]
    
    noise_r = K.random_normal(K.shape(real), mean=0, stddev=noise_std)
    noise_i = K.random_normal(K.shape(imag), mean=0, stddev=noise_std)
    
    real = real + noise_r
    imag = imag + noise_i   
    
    Y = K.stack([real, imag, H_R_est, H_I_est], axis=2)
    Y = tf.Session().run(Y)    
    
    return Y

"""======= build MC-AE autoecoder model =============="""
# encoder
X = Input(shape=(M,))
enc = Dense(2*N, use_bias=True, activation=act_enc, init=ini)(X) # be careful
if norm_type == 0:
    enc = Lambda(lambda x: np.sqrt(N) * K.l2_normalize(x, axis=1))(enc)
else:
    enc = Lambda(lambda x: x/tf.sqrt(2*tf.reduce_mean(tf.square(x))))(enc) 
Z = Reshape((-1,2))(enc) # output of encoder or transmitted vector

# Y is input of decoder, Y may include received signal y=hx+n and channel h
Y = Lambda(lambda x: channel(x))(Z)

# decoder
n_hidden_layer_dec = len(hidden_layer_dec)
dec = Flatten()(Y)
for n in range(n_hidden_layer_dec):
    dec = Dense(hidden_layer_dec[n], activation=act_dec, init=ini)(dec)
    
X_hat = Dense(M, activation='softmax', init=ini)(dec) # estimate of X

# model encoder and decoder
AE = Model(X,X_hat)
encoder = Model(X,Z)

X_enc = Input(shape=(N,4,))
deco = AE.layers[-n_hidden_layer_dec-2](X_enc) # first layer of decoder
for n in range(n_hidden_layer_dec+1): # hidden and last layers of decoder
    deco = AE.layers[-n_hidden_layer_dec-1+n](deco)
decoder = Model(X_enc, deco)

# training or loading pre-trained model
if not pre_trained:
    train_data = generate_one_hot_vectors(M, train_size, get_label=False)
    AE.compile(optimizer=Adam(lr=l_rate), loss=f_loss) 
    AE.fit(train_data, train_data, epochs=n_epoch, batch_size=b_size, verbose=2)
    encoder.save_weights('./models/encoder_N4M16_.h5')
    decoder.save_weights('./models/decoder_N4M16_.h5')
else:
    encoder.load_weights('./models/encoder_N4M16.h5')
    decoder.load_weights('./models/decoder_N4M16.h5')
    

# Coding gain calculation
test_label = np.arange(M)
test_data = []
for i in test_label:
    temp = np.zeros(M)
    temp[i] = 1
    test_data.append(temp)
test_data = np.array(test_data)
U = encoder.predict(test_data)   
V = U.view(dtype=np.complex64)  # learned transmitted codewords
# print(V)
print('Coding gain = ', calculate_coding_gain(U))

# test data
test_data, test_label = generate_one_hot_vectors(M, test_size, get_label=True)
test_bit = (((test_label[:,None] & (1 << np.arange(m)))) > 0).astype(int)

# BLER calculation and plot
print('\nPerfect CSI BLER')
EbNodB_range = list(np.linspace(0, 25, 6))
BER = [None] * len(EbNodB_range)
BLER = [None] * len(EbNodB_range)
for n in range(0, len(EbNodB_range)):
    EbNo = 10 ** (EbNodB_range[n] / 10.0)
    noise_std = np.sqrt(1/(2*R*EbNo))
    
    no_errors = 0    
    Z = encoder.predict(test_data)    
    Y = channel_test(Z,noise_std,test_size,imperfect_channel=False)
    X_hat = decoder.predict(Y)
    pred_output = np.argmax(X_hat, axis=1)
    
    re_bit = (((pred_output[:, None] & (1 << np.arange(m)))) > 0).astype(int)
    bit_errors = (re_bit != test_bit).sum()
    BER[n] = bit_errors / test_size / m
    
    block_errors = (pred_output != test_label)
    block_errors = block_errors.astype(int).sum()
    BLER[n] = block_errors / test_size
    print('SNR:', EbNodB_range[n], 'BER:', BER[n], 'BLER:', BLER[n])
    
bler = BLER

# BLER calculation and plot for imperfect CSI
print('\nImperfect CSI BLER')
EbNodB_range = list(np.linspace(0, 25, 6))
BER = [None] * len(EbNodB_range)
BLER = [None] * len(EbNodB_range)
for n in range(0, len(EbNodB_range)):
    EbNo = 10 ** (EbNodB_range[n] / 10.0)
    noise_std = np.sqrt(1 / (2 * R * EbNo))

    no_errors = 0
    Z = encoder.predict(test_data)
    Y = channel_test(Z, noise_std, test_size, imperfect_channel=True)
    X_hat = decoder.predict(Y)
    pred_output = np.argmax(X_hat, axis=1)
    no_errors = (pred_output != test_label)
    no_errors = no_errors.astype(int).sum()
    BLER[n] = no_errors / test_size
    print('SNR:', EbNodB_range[n], 'BLER:', BLER[n])
    
bler_im = BLER
# plot BLER
plt.plot(EbNodB_range, bler,'bo-', label='MC-AE under perfect CSI')
plt.plot(EbNodB_range, bler_im,'r+-', label='MC-AE under imperfect CSI')
plt.yscale('log')
plt.xlabel('SNR (dB)')
plt.ylabel('BLER')
plt.title('MC-AE with (M,N)=('+str(N)+','+str(M)+')')
plt.legend()
plt.grid()
plt.show()
 
# plot learned MC-AE constellations   
n1=0
n2=1
n3=2
n4=3
fig = plt.figure(figsize=(4,4))

if (N==4):
    plt.plot(U[:,n1,0], U[:,n1,1], "b.")
    plt.plot(U[:,n2,0], U[:,n2,1], "r*")
    plt.plot(U[:,n3,0], U[:,n3,1], "go")
    plt.plot(U[:,n4,0], U[:,n4,1], "k+")
elif N==2:
    plt.plot(U[:,n1,0], U[:,n1,1], "b.")
    plt.plot(U[:,n2,0], U[:,n2,1], "r*")
else:
    plt.plot(U[:,n1,0], U[:,n1,1], "b.")
    plt.plot(U[:,n2,0], U[:,n2,1], "r*")
    plt.plot(U[:,n3,0], U[:,n3,1], "go")
plt.grid(True)
plt.title('MC-AE constellation')
plt.show()

# save learned codewords
import scipy.io 
scipy.io.savemat('.\plot_constellation\cons_data\cons_xx.mat', dict(U=U))