import numpy as np
import pandas as pd

def generate_data(n, p, p2, K, H, location1, scale1, location2 = None, scale2 = None, random_state = None):
    ''''
    n is the number os objects
    p is the number of variables
    p2 is the number of irrelevant variables. (If there are irrelevant variables, then the number of relevant variables is p - p2)
    K is the number of object clusters
    H is the number of variables clusters
    location1 is a matrix K x H with the means of the gaussian distribution to generate each block
    scale1 is a matrix K x H with the standard deviations of the gaussian distribution to generate each block
    location2 is a vector of means of size p2 for multivariate normal distribution used to generate the irrelevant variables
    scale2  is a scale or a vector of stanrdart deviations for multivariate normal distribution used to generate the irrelevante variables
    random_state is the seed for generate the data, i.e controls randomness to generate the data
    
    PS: if p2 is zero, then location2 and scale2 muste be None
    '''



    X = pd.DataFrame(np.zeros((n,p)))
    p1 = p - p2
    n_blocks = K*H
    n_location = np.size(location1) 
    n_scale = np.size(scale1)

    if n_blocks != n_location or n_blocks != n_scale:
        return "Error: The number of parameters must be equal to the number of the blocks"

    np.random.seed(random_state)
    pi = np.ones(K)/K
    nb_object = np.random.multinomial(n,pi)
    lk = np.cumsum(nb_object,dtype='int64')
    lk = np.concatenate((np.array([0]),lk))
    
    ro = np.ones(H)/H
    nb_features = np.random.multinomial(p1,ro)
    lh = np.cumsum(nb_features,dtype='int64')
    lh = np.concatenate((np.array([0]),lh))
   
    labels = np.array([])
    for k in range(K):
        labels = np.concatenate((labels,np.repeat(k,nb_object[k])))
        ik = np.arange(lk[k],lk[k+1])
        for h in range(H):
            jh = np.arange(lh[h],lh[h+1])
            X.iloc[ik,jh]= np.random.normal(loc = location1[k,h],scale = scale1[k,h], size = ( nb_object[k], nb_features[h] )) 
    if p2 != 0:
        if p2 != len(location2):
            return "Error: The lenght of the vector of the averages must be equal to the number of irrelevant features"

        if np.size(scale2) == 1:
            scale2 = np.identity(p2) * scale2

        if np.size(scale2) == p2:
            scale2 = np.diag(scale2)

        X.iloc[:,p1:] = np.random.multivariate_normal(mean = location2, cov = scale2, size = n)

    return X.to_numpy(),labels.astype('int64')
    