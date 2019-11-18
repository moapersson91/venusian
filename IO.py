import numpy as np
import sys


def load_data(file):


    try:
        data = np.load(file).item()  # Load the data file
        
    except:
        # save np.load
        np_load_old = np.load

        # modify the default parameters of np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        data = np.load(file).item()  # Load the data file.

        # restore np.load for future normal usage
        np.load = np_load_old

    return data


def get_data(filepath = 'processed_data/', filename='venus_specData'):

    # load data
    specData = load_data('{}{}.npy'.format(filepath, filename))

    return specData
    
    
def save_file(data, filepath = '', filename = 'data'):
    import pickle
    with open('{}{}.pickle'.format(filepath, filename), 'wb') as f:
        pickle.dump(data, f)

def read_file(filepath = '', filename = 'data'):
    import pickle
    with open('{}{}.pickle'.format(filepath, filename), 'rb') as f:
        data = pickle.load(f)
    return data
