import numpy as np
import math
import matplotlib.pyplot as plt

import fir1
from scipy import signal
from scipy.ndimage import convolve
from scipy.linalg import fractional_matrix_power

def correlationMatrix (I, J, dVert, dHor):
    nPoints = I*J
    p = np.zeros([nPoints, nPoints])

    for m in range(0,nPoints):
        j = float(int(((m)/I) - 0.5))
        i = m
        if i == -1:
            i = I-1
        for n in range(0,nPoints):
            k = float(int(((n)/I) - 0.5))
            l = n
            if l == -1:
                l = I-1
            p[m,n] = math.exp(-math.sqrt(((i-l)/(dVert))**2))*math.exp(-math.sqrt(((j-k)/(dHor))**2))
    return p

def lowPassFilter2(signalp, Ts, Ncoef, cutFreq):
    single_dimensional = len(signalp.shape) == 1
    if single_dimensional:
        signalp = signalp.reshape((-1, 1))
    Ts = Ts * 1e-3
    impFiltered = np.zeros([signalp.shape[0], signalp.shape[1]])
    for j in range(0, signalp.shape[1]):
        impFiltered[:, j], h = lowPassFilter1(signalp[:, j], Ts, Ncoef, cutFreq)
    if single_dimensional:
        return impFiltered[:, 0]
    else:
        return impFiltered
        
def lowPassFilter1(signalp, Ts, Ncoef, cutFreq):
    Fnorm = cutFreq * 2 * Ts
    h = signal.firwin(Ncoef + 1, Fnorm)
    signalMod = np.zeros(2 * Ncoef + signalp.size + 1)
    signalMod[0:Ncoef] = signalp[0]
    signalMod[Ncoef : Ncoef + signalp.size] = signalp
    signalMod[Ncoef + signalp.size : 2 * Ncoef + signalp.size + 1] = signalp[-1]
    impFiltered = signal.lfilter(h, 1, signalMod)
    impFiltered = impFiltered[(int)(3 * Ncoef / 2) : (int)(signalp.size + (3 * Ncoef / 2))]
    return impFiltered, h

def acoustic_foward_matrix(wavelet,n):
    D = np.zeros([n-1,n])
    for i in range(0,n-1):
        D[i,i] = -1
        D[i,i+1] = 1
    
    n_w = wavelet.size
    nl2 = int((n_w-1)/2)
    C = convmtx(wavelet,n-1)
    C = C[:,nl2: - nl2]
    G = 0.5*C.dot(D)
    return G

def convmtx(v, n):
    N = len(v) + 2*n - 2
    xpad = np.concatenate([np.zeros(n-1), v.flatten(), np.zeros(n-1)])
    X = np.zeros((len(v)+n-1, n))
    # Construct X column by column
    for i in range(0,n):
        X[:,i] = xpad[n-i-1:N-i]
    
    return X.transpose()

def covariance_matrix_exp(sgm2, L, order):
    sgm = np.sqrt(sgm2)
    I = len(sgm)

    x = np.linspace(0,I, num=I)
    y = np.linspace(0,I, num=I)

    X, Y = np.meshgrid(x,y)

    covar = np.exp(-(np.abs(X[:]-Y[:])/L)**order)
    covar = covar.reshape(I,I)
    SGM = np.diag(sgm.reshape(-1))
    covar = SGM.dot(covar.dot(SGM))
    return covar

def get_forward_model(m_ref, a, b, nl_level):
    if nl_level < 1:
        nl_level = 1
    return a*np.exp(-m_ref*nl_level)+b

def facies_forward_model(facies_sample, PRIOR, G, variance_factor=0.1):
    mu, C = construct_prior_facies_acoustic(PRIOR, facies_sample, 2.5)
    log_imp = np.random.multivariate_normal(mu.flatten(),variance_factor*C)
    return mu, log_imp, G.dot(log_imp)

def construct_prior_facies_acoustic(PRIOR, facies_sample, L):
    I = len(facies_sample)
    mask_byfacies = np.zeros([I,I])    
    var_ = np.zeros([I,1])    
    A = covariance_matrix_exp(np.ones([I,1]),L,1)
    mu = np.zeros([len(facies_sample),1])
    var_ = np.zeros([len(facies_sample),1])
    mask_byfacies = np.zeros([len(facies_sample),len(facies_sample)])
    for facies in range(0,len(PRIOR)):
        index = np.where(facies_sample==facies)
        mu[index,0] = PRIOR[facies]['MU'][3]
        var_[index] = np.sqrt(PRIOR[facies]['C'][3,3])
        mask_byfacies[index,index] = 1
    C = np.diag(var_.reshape(-1)).dot(A).dot(np.diag(var_.reshape(-1)))
    mask = mask_byfacies
    C = C.dot(mask)
    return mu, C

def simulate_markov_chain(P, n, initial_facies, nsims, prob_map = None):
    """Simulate Markov Chain
    
    Arguments:
        P {matrix} -- Transition matrix
        n {integer} -- Size of the simulation
        initial_facies {array} -- Initial facies of the chain to start the simulation
        nsims {integer} -- Number of simulations
        prob_map {optional} -- Pointwise prior probability, usually comes from the Bayesian inference/classification
    """
    n_facies = P.shape[1]

    simulation = np.zeros([n, nsims])

    simulation[0,:] = initial_facies

    probabilities = []

    if prob_map == None:
        for j in range(0,nsims):
            for i in range(1,n):
                facies = np.array(np.where(np.random.rand() < np.cumsum(P[int(simulation[i-1,j]),:])))
                simulation[i,j] = facies[0,0]
    else:
        for j in range(0,nsims):
            for i in range(1,n):
                probabilities = P[simulation[i-1,j],:].dot(np.reshape(prob_map[i,0,:], 1, n_facies))
                probabilities = probabilities/np.sum(probabilities)
                facies = np.where(np.random.rand() < np.cumsum(probabilities))
                simulation[i,j] = facies[0]
    return simulation

def simulate_markov_2Dchain(Ph, Pv, prior_map, initial_facies = None):
    n_facies = Pv.shape[1]

    I = prior_map.shape[0]
    J = prior_map.shape[1]

    simulation = np.zeros([I, J]) - 1.0

    if initial_facies is None:
        initial_facies = np.around(np.random.randint(0,n_facies))
        simulation[0,:] = simulate_markov_chain(Ph,J,initial_facies,0)
    else:
        simulation[0,:] = initial_facies

    probabilities = []

    for i in range(1,I):
        random_path = np.random.permutation(J)
        
        # Simulate the first facies at a random position conditioned to the top
        i1 = random_path[0]
        probabilities = Pv[int(simulation[i-1,i1])] * prior_map[i,i1]
        probabilities = probabilities/np.sum(probabilities)
        facies = np.where(np.random.rand() < np.cumsum(probabilities))
        simulation[i,i1] = facies[0][0]

        # Simulate the second facies at a random position conditioned to the top and to the first
        i2 = random_path[1]
        Pt = np.linalg.matrix_power(Ph, np.absolute(i2-i1))
        probabilities = Pt[int(simulation[i,i1])]* Pv[int(simulation[i-1,i2])] * prior_map[i,i2]
        probabilities = probabilities/np.sum(probabilities)
        facies = np.where(np.random.rand() < np.cumsum(probabilities))
        simulation[i,i2] = facies[0][0]

        iter = random_path[2:]

        for j in iter:
            idxs_simulated = np.where(simulation[i] != -1.0)[0]
            distances = np.abs(j-idxs_simulated + 0.1).flatten()
            minimos = np.sort(distances)
            minimos = minimos[0:2]
            
            i1 = np.where(distances == minimos[0])[0][0]
            P1 = fractional_matrix_power(Ph,minimos[0])
            i2 = np.where(distances == minimos[1])[0][0]
            P2 = fractional_matrix_power(Ph,minimos[1])

            probabilities = P1[int(simulation[i,idxs_simulated[i1]])] * P2[int(simulation[i,idxs_simulated[i2]])] * Pv[int(simulation[i-1,j])] * prior_map[i,j]
            probabilities = probabilities/np.sum(probabilities)
            facies = np.where(np.random.rand() < np.cumsum(probabilities))
            simulation[i,j] = np.array(facies)[0][0]
    return simulation

