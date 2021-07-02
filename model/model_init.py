import sys
import os
sys.path.append(os.path.abspath('../dynamo'))
sys.path.append(os.path.abspath('../data_processing'))

from film_metadata_read import readAllPlots, readTitlesFromInds, readEncodingFromID, readAllEncodings
from word2vec_read import readSingleWord, readBatchWords
import omdb_retrieval