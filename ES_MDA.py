import numpy as np
import math
from utils import facies_forward_model
from multiprocessing import Pool

"""
ES-MDA with Deep Learning based on https://github.com/smitharauco/GeoFacies_DL
"""

def ES_MDA(num_ens,m_ens,Z,prod_ens,alpha,CD,corr,numsave=2):
    varn=1-1/math.pow(10,numsave)
    # Initial Variavel 
    # Forecast step
    yf = m_ens                        # Non linear forward model 
    df = prod_ens                     # Observation Model
    numsave
    ym = np.array(yf.mean(axis=1))    # Mean of the y_f
    dm = np.array(df.mean(axis=1))    # Mean of the d_f
    ym=ym.reshape(ym.shape[0],1)    
    dm=dm.reshape(dm.shape[0],1)    
    dmf = yf - ym
    ddf = df - dm
    
    #print('Covariance matrixes')
    Cmd_f = (np.dot(dmf,ddf.T))/(num_ens-1);  # The cros-covariance matrix
    Cdd_f = (np.dot(ddf,ddf.T))/(num_ens-1);  # The auto covariance of predicted data
    
    # Perturb the vector of observations
    R = np.linalg.cholesky(CD) #Matriz triangular inferior
    U = R.T   #Matriz R transposta
    p , w =np.linalg.eig(CD)
    
    aux = np.repeat(Z,num_ens,axis=1)
    mean = 0*(Z.T)

    noise=np.random.multivariate_normal(mean[0], np.eye(len(Z)), num_ens).T
    d_obs = aux+math.sqrt(alpha)*np.dot(U,noise)  
    #d_obs = aux+math.sqrt(alpha)*np.random.normal(Z.shape)  
    
    # Analysis step
    u, s, vh = np.linalg.svd(Cdd_f+alpha*CD); v = vh.T
    diagonal = s
    for i in range(len(diagonal)):
        if (sum(diagonal[0:i+1]))/(sum(diagonal)) > varn:
            diagonal = diagonal[0:i+1]
            break
    
    u=u[:,0:i+1]
    v=v[:,0:i+1]
    ss = np.diag(diagonal**(-1))
    K=np.dot(Cmd_f,(np.dot(np.dot(v,ss),(u.T))))
    # Use Kalman covariance
    if len(corr)>0:
        K = corr*K
        
    ya = yf + (np.dot(K,(d_obs-df)))
    m_ens = ya
    return m_ens

def facies_forward_model_2D(facies, PRIOR, G, v_fact):
  seismics = []
  impedances = []
  for j in range(0,facies.shape[0]):
    mu, log_imp, seismic = facies_forward_model(facies[:,j], PRIOR, G, v_fact)
    seismics.append(seismic)
    impedances.append(log_imp)

  seismics = np.array(seismics).transpose()
  impedances = np.array(impedances).transpose()
  return seismics, impedances

def forward_model(facies, dim_shape, PRIOR, G, v_fact):
    seis_exp_tmp, imp_exp_tmp = facies_forward_model_2D(facies.reshape(dim_shape), PRIOR, G, v_fact)
    return seis_exp_tmp, imp_exp_tmp

def get_obs_sim(m_x,dim_shape,PRIOR, G, network):
    v_fact = 0.1
    seis = []
    imps = []
    lat = []
    if type(network.latent_dim_shape) is not int:
        lat = m_x.reshape([m_x.shape[0]]+list(network.latent_dim_shape))
    else:
        lat = m_x.reshape([m_x.shape[0], network.latent_dim_shape])
    facies = network.decode(lat).numpy()

    with Pool(10) as pool:
        results = pool.starmap(forward_model, [(facies[e], dim_shape, PRIOR, G, v_fact) for e in range(facies.shape[0])])
    
    seis, imps = zip(*results)
    seis = np.array(seis).reshape([m_x.shape[0],-1])
    imps = np.array(imps).reshape([m_x.shape[0],-1])
    return seis, imps

def ES_MDA_DL(alp,Corr,obs,R,m_x,m_f,dim_shape,PRIOR,G,redeVAE):
    Alpha = np.ones((alp),dtype=int)*alp
    m_x_a = np.zeros(tuple([alp])+m_x.shape)
    m_f_a = np.zeros(tuple([alp])+m_f.shape)
    for i in range(len(Alpha)+1):
        Obs_sim,_ = get_obs_sim(m_x.T,dim_shape,PRIOR,G,redeVAE)
        Obs_sim = Obs_sim.T
        #mse = (abs(Obs_sim - np.repeat(obs,Obs_sim.shape[-1],axis=1)) ** 2).mean(axis=1).mean()
        print('Error iteration',i, ' : ', sum(sum(abs(Obs_sim-np.repeat(obs,Obs_sim.shape[-1],axis=1)))))
        if i < len(Alpha):
            m_x = m_x.reshape([-1,m_x.shape[-1]])
            m_f = m_f.reshape([-1,m_f.shape[-1]])
            m_x = ES_MDA(m_f.shape[1],m_x,obs,Obs_sim,Alpha[i],R,Corr,2)
            if type(redeVAE.latent_dim_shape) is not int:
                m_x = m_x.reshape(list(redeVAE.latent_dim_shape)[::-1] + [m_x.shape[-1]])
            else:
                m_x = m_x.reshape([redeVAE.latent_dim_shape,m_x.shape[-1]])
            m_f =redeVAE.decode(m_x.T).numpy().reshape([m_x.T.shape[0],-1]).T
            m_x_a[i] = m_x
            m_f_a[i] = m_f
    return m_f, m_x_a, m_f_a