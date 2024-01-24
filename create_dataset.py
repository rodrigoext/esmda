from datetime import datetime
from multiprocessing import Pool
import time
import h5py
import numpy as np
from utils import *
from scipy.ndimage.filters import gaussian_filter
from tqdm.contrib.concurrent import process_map

I = 48
J = 48
result = []

def generate_image(_):
    Pver = np.array(np.matrix('0.7 0.3 0 0; 0.3 0.7 0 0; 0.33 0.33 0.34 0; 0.15 0.15 0.15 0.55'))

    Phor = np.array(np.matrix('0.4 0.4 0.1 0.1; 0.4 0.4 0.1 0.1; 0.1 0.1 0.6 0.2; 0.1 0.1 0.2 0.6'))
    
    initial_facies = 3

    prior_map = np.ones([I, J, 4])
    simulation = simulate_markov_2Dchain(Phor, Pver, prior_map, initial_facies)
    ss = gaussian_filter(simulation, sigma=[0.5, 1.5])
    st = simulation == 3
    ss[st] = 3
    ss = np.round(gaussian_filter(ss, sigma=[0.5 , 1.5]))
    return np.round(ss)

# Apply
def create_data():
    num_images = 50000
    res = process_map(generate_image,range(0,num_images), max_workers=11)
    result.append(res)
    X_bk = np.asarray(result, dtype=np.ubyte)[0]
    print(X_bk.shape,X_bk[0])
    h5f = h5py.File('data_48x48_2D.h5', 'w')
    h5f.create_dataset('X', data=X_bk)
    h5f.close()

if __name__ == '__main__':
    start = datetime.now()
    create_data()
    print("End Time To Create Dataset:", (datetime.now() - start).total_seconds())
