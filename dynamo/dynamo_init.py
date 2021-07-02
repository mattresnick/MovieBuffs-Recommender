import sys
import os
sys.path.append(os.path.abspath('../model'))
sys.path.append(os.path.abspath('../data_processing'))

from Encoder import Encoder
import StrToModel, hparameters

import omdb_retrieval
import tmdb_retrieval

try:
    with open('../data_processing/apikey.txt','r') as f: key=f.read()
    with open('../data_processing/tmdb_key.txt','r') as f: tkey=f.read()
except:
    print ('No key files provided. API requests will fail.')
    key, tkey = '', ''