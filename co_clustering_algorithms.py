import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist
from scipy.stats.mstats import gmean


#----------------------------------------- Auxiliar functions -----------------------------------------# 

# Functions for all algorithms

def random_U(shape, random_state = None):
    
    if type(shape) == int:
        np.random.seed(random_state)
        h = np.random.random(shape)
        return h/h.sum()
    else:
        np.random.seed(random_state)
        h = np.random.random(shape)
        return h/h.sum(axis= 1).reshape(-1,1) 

def D_adaptive(D,W):

    if np.ndim(W) == 1:
        return D * W
    else:
        Da = np.zeros_like(D)
        K = D.shape[0]
        for k in range(K):
            Da[k] = D[k] * W[k]
        return Da


# Functions for the state-of-the-art algorithms (FDK and WFDK)

def dist_array(X, G):
    '''
    X is the dataset
    G is the prototypes of the co-clusters
    sig2 is the parameters of the gaussian kernel
    '''
    N, P = X.shape
    K, H = G.shape
    D = np.zeros((K, H, N, P))
    for k in range(K):
        for h in range(H):
            D[k,h] = (X - G[k,h]) ** 2    
    return D

def prototypes_fuzzy(X, Um, Vn, G, lowest_denominator):
    '''
    X is the dataset 
    Um is the fuzzy matrix of the objects into K cluster raised to the m
    Vn is the fuzzy matrix of the variables into H cluster raised to the n
    G is the current prototype matrix
    ld is the lowest denominator, to avoid indeterminacy.
    '''
    K,H = G.shape
    for k in range(K):
        uk = (Um[:,k]).reshape((-1,1))
        for h in range(H):
            vh = (Vn[:,h]).reshape((1,-1))
            wh = (uk * np.full_like(X,1) * vh)
            if wh.sum() > lowest_denominator:
                G[k,h] = np.average(a = X, weights = wh )            
    return G

def prototypes_fuzzy_adaptive(X, W, Um, Vn, G, lowest_denominator):
    '''
    X is the dataset 
    W is the weights of the variables
    Um is the fuzzy matrix of the objects into K cluster raised to the m
    Vn is the fuzzy matrix of the variables into H cluster raised to the n
    G is the current prototype matrix
    '''
    K,H = G.shape
    P = X.shape[1]

    if np.ndim(W) == 1:
        W = np.tile(W,K).reshape((K,P))

    for k in range(K):
        uk = (Um[:,k]).reshape((-1,1))
        wk = (W[k]).reshape((1,-1))
        for h in range(H):
            vh = (Vn[:,h]).reshape((1,-1))
            wh = (uk * np.full_like(X,1) * vh * wk)
            if wh.sum() > lowest_denominator:
                G[k,h] = np.average(a = X, weights = wh )            
    return G

def get_weights_global_sum(D, Um, Vn, beta, ld):
    '''
    D is the array of distance between X and the co-closters
    Um is the fuzzy matrix of the objects into K cluster raised to the m
    Vn is the fuzzy matrix of the variables into H cluster raised to the n
    beta is the parameter for the computation of the withgs with sum constraint
    ld is the lowest denominator, to avoid indeterminacy
    '''
    K,H = D.shape[:2]
    P = D.shape[3]
    Dk = np.zeros((K,P))
    for k in range(K):
        uk = (Um[:,k]).reshape((-1,1))
        Dhp = np.zeros((H,P))
        for h in range(H):
            vh = (Vn[:,h]).reshape((1,-1))
            Dhp[h] = (uk * D[k,h] * vh).sum(axis = 0)
        Dk[k] = Dhp.sum(axis = 0)
    Dj = Dk.sum(axis = 0)
    idj = np.where(Dj > ld)[0]
    nj = len(idj)
    W = np.zeros(P)
    if nj == P:
        for j in range(P):
            ratio = (Dj[j]/Dj)**( 1/(beta-1) )
            W[j] =  ratio.sum() ** (-1)
    else:
        Dju = Dj[idj]
        Wu = np.zeros(nj)
        for a in range(nj):
            ratio = (Dju[a]/Dju)**( 1/(beta-1) )
            Wu[a] =  ratio.sum() ** (-1)
        W[idj] = Wu
    return W

def getU_fuzzy(D, U, V, m, n, ld):
    '''
    D is the array of the distances between the dataset and the prototype matrix
    V is the partiton of variables into H cluster
    m and n are the fuzzyness parameters
    '''
    exponent = (-1)/(m-1)
    K,N = D.shape[0], D.shape[2]
    for i in range(N):
        Di = D[:,:,i,:] 
        Dvi = ((V**n).T)*Di
        Dik = Dvi.sum(axis = (1,2))

        idx = np.where(Dik < ld)[0]
        nzeros = len(idx)
        inv_dk = Dik ** exponent
        if inv_dk.sum() > 0.0:
            idx_inf =  np.where(np.isinf(inv_dk))[0]
            idx = np.where(~np.isinf(inv_dk))[0]
            n_inf = len(idx_inf)
            if n_inf == 0:
                U[i] = inv_dk/inv_dk.sum()
            else:
                sum_previous = U[i,idx_inf].sum()
                inv_dk_new = inv_dk[idx]
                if inv_dk_new.sum() > 0.0:
                    U[i,idx] = (1 - sum_previous) * (inv_dk_new/inv_dk_new.sum())
    return U 

def getV_fuzzy(D, U, V, m, n, ld):
    '''
    D is the array of the distances between the dataset and the prototype matrix
    V is the partiton of variables into H cluster
    m and n are the fuzzyness parameters
    '''
    exponent = (-1)/(n-1)
    H,P = D.shape[1], D.shape[3]
    Dm = np.moveaxis(D,0,1)
    for j in range(P):
        Dj = Dm[:,:,:,j] 
        Duj = ((U**m).T)*Dj
        Djh = Duj.sum(axis = (1,2))

        idx = np.where( Djh < ld)[0]
        inv_dh = Djh ** exponent
        if inv_dh.sum() > 0.0:
            idx_inf =  np.where(np.isinf(inv_dh))[0]
            idx = np.where(~np.isinf(inv_dh))[0]
            n_inf = len(idx_inf)
            if n_inf == 0:
                V[j] = inv_dh/inv_dh.sum()
            else:
                sum_previous = V[j,idx_inf].sum()
                inv_dk_new = inv_dh[idx]
                if inv_dk_new.sum() > 0.0:
                    V[j,idx] = (1 - sum_previous) * (inv_dk_new/inv_dk_new.sum())
    return V

def computeJ_fuzzy(D, Um, Vn):
    '''
    D is the array of the distance between the dataset and the prototype matrix
    Um is the fuzzy matrix of the objects into K cluster raised to the m
    Vn is the fuzzy matrix of the variables into H cluster raised to the n
    '''
    K,H = D.shape[:2]
    Jc = 0
    for k in range(K):
        uk = (Um[:,k]).reshape((-1,1))
        for h in range(H):
            vh = (Vn[:,h]).reshape((1,-1))
            Dkh = uk * D[k,h] * vh
            Jc += Dkh.sum()
    return Jc


# Functions for the proposed algorithms (GKFDK and WGKFDK)

def gaussian_kernel_array(X, G, sig2):
    '''
    X is the dataset
    G is the prototypes of the co-clusters
    sig2 is the hiperparameter of the gaussian kernel
    '''
    n, p = X.shape
    K, H = G.shape
    const = 1/(2*sig2)
    KMs = np.zeros((K, H, n, p))
    for k in range(K):
        for h in range(H):
            KMs[k,h] = np.exp( -const * (X - G[k,h]) ** 2 )     
    return KMs

def initial_prototypes(X, Um, Vn):
    '''
    X is the dataset 
    KMs is the array of kernel between X and G
    Um is the fuzzy matrix of the objects into K cluster raised to the m
    Vn is the fuzzy matrix of the variables into H cluster raised to the n
    ld is the lowest denominator, to avoid indeterminacy.
    '''
    K, H = Um.shape[1], Vn.shape[1]
    G = np.zeros((K,H))
    for k in range(K):
        uk = (Um[:,k]).reshape((-1,1))
        for h in range(H):
            vh = (Vn[:,h]).reshape((1,-1))
            wh = (uk * np.full_like(X,1) * vh)
            G[k,h] = np.average(a = X, weights = wh )            
    return G

def prototypes(X, KMs, Um, Vn, G, ld):
    '''
    X is the dataset 
    KMs is the array of kernel between X and G
    W is the weights of the variables
    Um is the fuzzy matrix of the objects into K cluster raised to the m
    Vn is the fuzzy matrix of the variables into H cluster raised to the n
    G is the current prototype matrix
    ld is the lowest denominator, to avoid indeterminacy.
    '''
    K,H = G.shape
    P = X.shape[1]
    
   
    for k in range(K):
        uk = (Um[:,k]).reshape((-1,1))
        for h in range(H):
            vh = (Vn[:,h]).reshape((1,-1))
            KMkh = uk * KMs[k,h]* vh 
            if KMkh.sum() > ld:
                G[k,h] = np.average(a = X, weights = KMkh )            
    return G

def prototypes_adaptive(X, KMs, W, Um, Vn, G, ld):
    '''
    X is the dataset 
    KMs is the array of kernel between X and G
    W is the weights of the variables
    Um is the fuzzy matrix of the objects into K cluster raised to the m
    Vn is the fuzzy matrix of the variables into H cluster raised to the n
    G is the current prototype matrix
    ld is the lowest denominator, to avoid indeterminacy.
    '''
    K,H = G.shape
    P = X.shape[1]
    
    if np.ndim(W) == 1:
        W = np.tile(W,K).reshape((K,P))
   
    for k in range(K):
        uk = (Um[:,k]).reshape((-1,1))
        wk = W[k].reshape((1,-1))
        for h in range(H):
            vh = (Vn[:,h]).reshape((1,-1))
            KMkh = uk * KMs[k,h]* vh * wk
            if KMkh.sum() > ld:
                G[k,h] = np.average(a = X, weights = KMkh )            
    return G

def get_weights_local_prod(D, Um, Vn, W, ld):
    '''
    D is the array of distance between X and the co-closters
    Um is the fuzzy matrix of the objects into K cluster raised to the m
    Vn is the fuzzy matrix of the variables into H cluster raised to the n
    W is the weights of the variables at previous iterations
    ld is the lowest denominator, to avoid indeterminacy.
    '''
    vhn = Vn.sum(axis = 0)
    K,H,N,P = D.shape
    for k in range(K):
        uik = (Um[:,k]).reshape((-1,1))
        DN = np.zeros((H,P))
        for h in range(H):
            vjh = (Vn[:,h])
            DN[h] = (uik * D[k,h]).sum(axis = 0) * (vjh/vhn[h])
        DN = np.where(np.isinf(DN),0,DN)
        DHN = DN.sum(axis = 0)
        if (DHN > ld).all():
            W[k] =  gmean(DHN)/DHN
        else:
            jless = np.where(DHN<= ld)[0]
            jupper = np.where(DHN > ld)[0]
            nless = len(jless)
            if nless == P:
                continue
            const = (1/np.prod(W[k][jless]))**(1/(P-nless))
            sum_upper = DHN[jupper]
            W[k][jupper] = (const*gmean(sum_upper)/sum_upper)
    return W

def getU(D, U, V, m, n, ld):
    '''
    D is the array of the distances between the dataset and the prototype matrix
    V is the partiton of variables into H cluster
    m and n are the fuzzyness parameters
    '''
    Um = U**m
    Vn = V**n
    Vnt = np.transpose(Vn)
    ukm = Um.sum(axis = 0)
    vhn = Vn.sum(axis = 0)
    exponent = (1/(m-1))
    K,N = D.shape[0], D.shape[2]
    for i in range(N):
        Di = D[:,:,i,:] 
        Dvi = ((Vnt*Di).sum(axis = 2)) * (1/vhn)
        Dvi = np.where(np.isinf(Dvi),0,Dvi)
        Dvi = np.where(np.isnan(Dvi),0,Dvi)
        Dik = Dvi.sum(axis = 1)
        #inv_dk = (ukm/Dik) ** exponent
        inv_dk = (1/Dik) ** exponent
        if inv_dk.sum() > 0.0:
            idx_inf =  np.where(np.isinf(inv_dk))[0]
            idx = np.where(~np.isinf(inv_dk))[0]
            n_inf = len(idx_inf)
            if n_inf == 0:
                U[i] = inv_dk/inv_dk.sum()
            else:
                sum_previous = U[i,idx_inf].sum()
                inv_dk_new = inv_dk[idx]
                if inv_dk_new.sum() > 0.0:
                    U[i,idx] = (1 - sum_previous) * (inv_dk_new/inv_dk_new.sum())
    return U 

def getV(D, U, V, m, n, ld):
    '''
    D is the array of the distances between the dataset and the prototype matrix
    V is the partiton of variables into H cluster
    m and n are the fuzzyness parameters
    '''
    Um = U**m
    Vn = V**n
    Umt = np.transpose(Um)
    ukm = Um.sum(axis = 0)
    vhn = Vn.sum(axis = 0)
    exponent = (1/(n-1))
    H,P = D.shape[1], D.shape[3]
    Dm = np.moveaxis(D,0,1)
    for j in range(P):
        Dj = Dm[:,:,:,j] 
        Duj = ((Umt*Dj).sum(axis=2)) * (1/ukm)
        Duj = np.where(np.isinf(Duj),0,Duj)
        Duj = np.where(np.isnan(Duj),0,Duj)
        Djh = Duj.sum(axis = 1)
        #inv_dh = (vhn/Djh) ** exponent
        inv_dh = (1/Djh) ** exponent
        if inv_dh.sum() > 0.0:
            idx_inf =  np.where(np.isinf(inv_dh))[0]
            idx = np.where(~np.isinf(inv_dh))[0]
            n_inf = len(idx_inf)
            if n_inf == 0:
                V[j] = inv_dh/inv_dh.sum()
            else:
                sum_previous = V[j,idx_inf].sum()
                inv_dk_new = inv_dh[idx]
                if inv_dk_new.sum() > 0.0:
                    V[j,idx] = (1 - sum_previous) * (inv_dk_new/inv_dk_new.sum())
    return V

def computeJ(D, Um, Vn):
    '''
    D is the array of the distance between the dataset and the prototype matrix
    Um is the fuzzy matrix of the objects into K cluster raised to the m
    Vn is the fuzzy matrix of the variables into H cluster raised to the n
    '''
    ukm = Um.sum(axis = 0)
    vhn = Vn.sum(axis = 0)
    K,H = D.shape[:2]
    Jc = 0
    for k in range(K):
        if ukm[k] != 0:
            uk = (Um[:,k]).reshape((-1,1))
            for h in range(H):
                if vhn[h] != 0:
                    vh = (Vn[:,h]).reshape((1,-1))
                    const = (1/ukm[k]) * (1/vhn[h])
                    if ~np.isinf(const):
                        Dkh = const * ( (uk * D[k,h] * vh).sum() )
                        Jc += Dkh.sum()
    return Jc

#----------------------------------------- Main functions -----------------------------------------#

def FDK(X, K, H, m, n, random_state = None, epsilon = 10**(-10), T = 100, lowest_denominator = 10**(-100)):
    '''
    X is a dataset
    K is the number of cluster for the objetcs
    H is the numer of cluster for the variables
    m and n are the fuzzyness pararmeters
    epsilon is the tolerance for J
    T is the maximum iteration number 
    ld is the lowest allowed denominator. It's used to avoid indeterminations
    '''
    if type(X) != np.ndarray:
        names = X.columns
        X = X.to_numpy()
    N,P = X.shape

    # Inicialization step
    np.random.seed(random_state)
    G = np.random.choice(np.unique(X), K*H, replace= False).reshape((K,H))
    U = random_U((N,K),random_state)
    V = random_U((P,H),random_state)
    Um = U**m
    Vn = V**n
    J = 0
    Jlist = []

    # iterative step
    t = 1
    while True:
        G = prototypes_fuzzy(X, Um, Vn, G,lowest_denominator)
        D = dist_array(X, G)
        U = getU_fuzzy(D, U, V,  m, n, ld = lowest_denominator)
        V = getV_fuzzy(D, U, V,  m, n, ld = lowest_denominator)
        Um = U**m
        Vn = V**n
        Jcurr = J
        J = computeJ_fuzzy(D, Um, Vn)
        Jlist.append(J)

        if np.abs(Jcurr - J) < epsilon or t > T:
            break
        else:
            t = t + 1

    # organize output
    knames = list()
    for k in range(K):
        knames.append( 'k=' + str(k+1))

    hnames = list()
    for h in range(H):
        hnames.append( 'h=' + str(h+1))

    G = pd.DataFrame(G, index = knames, columns = hnames)
    U = pd.DataFrame(U, index = list(range(N)),columns = knames)
    V = pd.DataFrame(V, index = list(range(P)),columns = hnames)
    return {'G':G, 'U':U, 'V':V, 'J':J, 'Jlist':Jlist, 't':t}

def WFDK(X, K, H, m, n, beta, random_state = None, epsilon = 10**(-10), T = 100, lowest_denominator = 10**(-100)):

    '''
    X is a dataset
    K is the number of objetc clusters
    H is the numer of variable clusters
    m and n are the fuzzyness pararmeters
    epsilon is the tolerance for J
    T is the maximum iteration number 
    ld is the lowest allowed denominator. It's used to avoid indeterminations
    '''
    if type(X) != np.ndarray:
        names = X.columns
        X = X.to_numpy()
    N,P = X.shape

    # Inicialization step
    W = random_U(P, random_state)
    U = random_U((N,K),random_state+1)
    V = random_U((P,H),random_state+2)
    Um = U**m
    Vn = V**n
    G = np.zeros((K,H))
    Wb = (W**beta)
    Jlist = []
    J = 0
    # iterative step
    t = 1
    while True:
        G = prototypes_fuzzy_adaptive(X, Wb, Um, Vn, G, lowest_denominator)
        D = dist_array(X, G)
        W = get_weights_global_sum(D, Um, Vn, beta,lowest_denominator)
        Wb = (W**beta)
        Da = D_adaptive(D,Wb)
        U = getU_fuzzy(Da, U, V,  m, n, ld = lowest_denominator)
        V = getV_fuzzy(Da, U, V,  m, n, ld = lowest_denominator)
        Um = U**m
        Vn = V**n
        Jcurr = J
        J = computeJ_fuzzy(Da, Um, Vn)
        Jlist.append(J)

        if (np.abs(Jcurr - J) < epsilon) or t > T:
            break
        else:
            t = t + 1

    # organize output
    knames = list()
    for k in range(K):
        knames.append( 'k=' + str(k+1))

    hnames = list()
    for h in range(H):
        hnames.append( 'h=' + str(h+1))

    G = pd.DataFrame(G, index = knames, columns = hnames)
    U = pd.DataFrame(U, index = list(range(N)),columns = knames)
    V = pd.DataFrame(V, index = list(range(P)),columns = hnames)
    return {'G':G,'W':W,'U':U, 'V':V, 'J':J, 'Jlist':Jlist, 't':t}

def GKFDK(X, K, H, m, n, sig2, random_state = None, epsilon = 10**(-10), T = 100, lowest_denominator = 10**(-100)):
    '''
    X is a dataset
    K is the number of cluster for the objetcs
    H is the numer of cluster for the variables
    m and n are the fuzzyness pararmeters
    sig2 is the hyperparameter of the gaussian kernel
    epsilon is the tolerance for J
    T is the maximum iteration number 
    ld is the lowest allowed denominator. It's used to avoid indeterminations
    '''
    if type(X) != np.ndarray:
        names = X.columns
        X = X.to_numpy()
    N,P = X.shape

    # Inicialization step
    U = random_U((N,K),random_state)
    V = random_U((P,H),random_state)
    Uinit = U.copy()
    Vinit = V.copy()
    Um = U**m
    Vn = V**n
    G = initial_prototypes(X, Um, Vn)
    KM = gaussian_kernel_array(X, G, sig2)
    D = (2-2*KM)
    J = 0
    Jlist = []

    # iterative step
    t = 1
    while True:
        U = getU(D, U, V, m, n, ld = lowest_denominator).copy()
        V = getV(D, U, V, m, n, ld = lowest_denominator).copy()
        Um = U**m
        Vn = V**n
        G = prototypes(X, KM, Um, Vn, G, lowest_denominator).copy()
        KM = gaussian_kernel_array(X, G, sig2).copy()
        D = (2 - 2*KM)
        Jcurr = J
        J = computeJ(D, Um, Vn).copy()
        Jlist.append(J)

        if np.abs(Jcurr - J) < epsilon or t > T:
            break
        else:
            t = t + 1

    if (Uinit == U).all() and (Vinit == V).all():
        print("U and V didn't change")
    elif (Uinit == U).all():
        print("U didn't change")
    elif (Vinit == V).all():
        print("V didn't change")
    else:
        print("all right")

    # organize output
    knames = list()
    for k in range(K):
        knames.append( 'k=' + str(k+1))

    hnames = list()
    for h in range(H):
        hnames.append( 'h=' + str(h+1))

    G = pd.DataFrame(G, index = knames, columns = hnames)
    U = pd.DataFrame(U, index = list(range(N)),columns = knames)
    V = pd.DataFrame(V, index = list(range(P)),columns = hnames)
    return {'G':G, 'U':U, 'V':V, 'J':J, 'Jlist':Jlist, 't':t}

def WGKFDK(X, K, H, m, n, sig2, random_state = None, epsilon = 10**(-10), T = 100, lowest_denominator = 10**(-100)):
    '''
    X is a dataset
    K is the number of cluster for the objetcs
    H is the numer of cluster for the variables
    m and n are the fuzzyness pararmeters
    sig2 is the hyperparameter of the gaussian kernel
    epsilon is the tolerance for J
    T is the maximum iteration number 
    ld is the lowest allowed denominator. It's used to avoid indeterminations
    '''
    if type(X) != np.ndarray:
        names = X.columns
        X = X.to_numpy()
    N,P = X.shape

    # Inicialization step
    U = random_U((N,K),random_state)
    V = random_U((P,H),random_state)
    Uinit = U.copy()
    Vinit = V.copy()
    Um = U**m
    Vn = V**n
    W = np.ones((K,P))
    G = initial_prototypes(X, Um, Vn)
    KM = gaussian_kernel_array(X, G, sig2)
    D = 2-2*KM
    Dj = D
    J = 0
    Jlist = []

    # iterative step
    t = 1
    while True:

        U = getU(Dj, U, V, m, n, ld = lowest_denominator)
        V = getV(Dj, U, V, m, n, ld = lowest_denominator)
        Um = (U ** m)
        Vn = (V ** n)
        W = get_weights_local_prod(D, Um, Vn, W, lowest_denominator)
        G = prototypes_adaptive(X, KM, W, Um, Vn, G, lowest_denominator)
        KM = gaussian_kernel_array(X, G, sig2)
        D = (2 - 2*KM)
        Dj = D_adaptive(D, W)
        Jcurr = J
        J = computeJ(Dj, Um, Vn).copy()
        Jlist.append(J)

        if np.abs(Jcurr - J) < epsilon or t > T:
            break
        else:
            t = t + 1

    if (Uinit == U).all() and (Vinit == V).all():
        print("U and V didn't change")
    elif (Uinit == U).all():
        print("U didn't change")
    elif (Vinit == V).all():
        print("V didn't change")
    else:
        print("all right")

    # organize output
    knames = list()
    for k in range(K):
        knames.append( 'k=' + str(k+1))

    hnames = list()
    for h in range(H):
        hnames.append( 'h=' + str(h+1))

    G = pd.DataFrame(G, index = knames, columns = hnames)
    W = pd.DataFrame(W, index = knames )
    U = pd.DataFrame(U, index = list(range(N)),columns = knames)
    V = pd.DataFrame(V, index = list(range(P)),columns = hnames)
    return {'G':G, 'W':W, 'U':U, 'V':V, 'J':J, 'Jlist':Jlist, 't':t}
