import numpy as np

# generate data, one-hot vector
def generate_one_hot_vectors(M, data_size, get_label=False):
    """
    Generate one hot vectors which are used as training data or testing data.
    
    Parameters:
    -----
        M: int, dimension of one-hot vectors, i.e, number of classes/categoraries
        data_size: int, number of one-hot vectors generated
    
    Return:
        data: shape(data_size,M) array
        
    """
    label = np.random.randint(M, size=data_size)
    data = []
    for i in label:
        temp = np.zeros(M)
        temp[i] = 1
        data.append(temp)
    data = np.array(data)
    
    if not get_label:
        return data
    else:
        return data, label
    
def calculate_coding_gain(U):
    V = U[:,:,0] + 1j*U[:,:,1]
    N = V.shape[1]   
    M = V.shape[0]
    Gd = N          # maximum diversity gain is the number of subcarriers
    Gc = 10000
    sum_Gc = 0
    
    for n1 in range(M):
        for n2 in range(n1+1,M):
            Y = np.prod(np.power(np.abs(V[n1]-V[n2]),2))
            sum_Gc = sum_Gc + 1/Y
            if (Y<Gc):
                Gc = Y           
    Gcc = np.power(sum_Gc/M/(2**(2*Gd)),-1/N)
    
    return Gcc
