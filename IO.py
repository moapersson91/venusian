import numpy as np
import sys


def load_data(file):


    try:
        data = np.load(file).item()  # Load the data file
        
    except:
        data = np.load(file, allow_pickle=True).item()  # Load the data file.

    return data


def get_data(filepath = 'processed_data', filename='venus_specData.npy'):

    # load data
    specData = load_data('{}/{}'.format(filepath, filename))

    return specData
    
    
def save_file(data, filepath = '', filename = 'data'):
    import pickle
    with open('{}/{}.pickle'.format(filepath, filename), 'wb') as f:
        pickle.dump(data, f)

def read_file(filepath = '', filename = 'data'):
    import pickle
    with open('{}/{}.pickle'.format(filepath, filename), 'rb') as f:
        data = pickle.load(f)
    return data
