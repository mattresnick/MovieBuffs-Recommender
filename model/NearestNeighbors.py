import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import boto3
from joblib import dump, load

from model_init import omdb_retrieval as o
from model_init import readTitlesFromInds, readEncodingFromID, readAllEncodings



def fitAndSaveNNModel(save_type='pickle'):
    
    enc_dict = readAllEncodings(ind=True)
    
    sorted_enc_dict = {k: v for k, v in sorted(enc_dict.items(), key=lambda x: x[0])}
    sorted_enc_list = list(sorted_enc_dict.values())
    
    # Fit nn model.
    nn_model = NearestNeighbors(n_neighbors=10)
    nn_model.fit(sorted_enc_list)
    
    if save_type=='pickle':
        with open('nnpickle', 'wb') as f:
            pickle.dump(nn_model, f)
    else:
        dump(nn_model, 'nnmodel.joblib') 
    



# Get recommendations, given a film title or sequence encoding.
def nearestTitles(model, title=None, encoding=None):
    if encoding is None:
        r_json = o.OMDbRequest(key='', title=title)
        
        if r_json['Response']=='False':
            fID = o.OMDbSearchReturnBest(key='', title=title)
        else:
            fID = r_json['imdbID']
        
        encoding = readEncodingFromID(fID)
    
    
    neighbor = model.kneighbors([encoding])[1]
    ind_list = [int(n) for n in neighbor[0]]
    
    titles = readTitlesFromInds(ind_list)
    
    return titles


def loadModel(filepath, loadtype='pickle'):
    if loadtype=='pickle':
        nn_model = pickle.load(open('nnpickle', 'rb'))
    else:
        nn_model = load('nnmodel')
    
    return nn_model


# Test inference, encoding or title.
with open('test_encoding.txt','rb') as f:
    test = np.frombuffer(f.read(), dtype=np.float64)

nn_model = loadModel('./saves/nnpickle')

print (nearestTitles(nn_model,title='Blade Runner'))




