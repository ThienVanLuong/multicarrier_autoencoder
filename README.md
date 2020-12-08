# Multicarrier Autoencoder
This is the first implementation of multicarrier autoencoder communication systems in the following paper:

T. V. Luong, Y. Ko and et. al, "Deep learning aided multicarrier systems", IEEE Trans. Wireless Commun., Nov., 2020 (Accepted). https://ieeexplore.ieee.org/abstract/document/9271932

Please use the files in folder ./plot_constellation to:
- Calculate coding gains for the proposed MC-AE and baselines whose constellation data is saved in folder ./plot_constellation/cons_data.
- Plot constellations for various MC-AE parameters N and M, as well as constellation of the baselines.
- Calculate the minimum Enclidean distance (MED) for learned codewords of MC-AE.

The BLER of MC-AE can be quickly obtained by running the pre-trained model stored in folder ./models. 

The main file of MC-AE is MC_AE.py. 

Requirements: 
- Tensorflow 1.15, Keras 2.0.8.
- Others: Numpy, scipy, matplotlib.

Note that further files for multi-user MC-AE systems, Emb-MC-AE, and the baselines will be updated soon. 
