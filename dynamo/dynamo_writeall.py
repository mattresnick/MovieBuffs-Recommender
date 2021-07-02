import gensim as g
import pandas as pd
import torch
import os

from word2vec_write import writeBatchWords
from film_metadata_write import writeBatchMetaData, swapEncodings

from dynamo_init import Encoder, StrToModel, hparameters

def word2vecWriteAll(last_stop=0, step=int(1e4), max_step=int(3e6)):
    w2v_path = 'GoogleNews-vectors-negative300.bin'
    model=g.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    
    # For batch/partitioned writes. 
    steps = range(last_stop+step, max_step, step)
    for w in steps:
        update = 'Step: {} - {}.'.format(str(w-step),str(w))
        print('\r{:30}'.format(update),end='')
        writeBatchWords(model, start=w-step, stop=w)
    
    writeBatchWords(model, start=steps[-1], stop=max_step)




def metadataWriteAll(titles_ids_plots,last_stop=0, step=int(1e4)):
    '''
    Parameters:
        titles_ids_plots: filepath to raw string data for all films
        last_stop: starting row index for batch write
        step: number of rows per write
    '''
    
    base_film_df = pd.read_csv(titles_ids_plots)
    base_film_df = base_film_df.rename(columns={'ID':'fID','Title':'title','Plot':'shortplot'})
    
    
    # Load in encoder.
    args = hparameters.model_args
    
    ae_model = Encoder(code_size=args['code_size'], 
                       em_len=args['em_len'], 
                       encoder_size=args['encoder_size'], 
                       n_dir=args['encoder_n_dir'])
    ae_model.load_state_dict(torch.load(os.path.abspath('../model')+'/encoder_statedict.pth'))
    ae_model = ae_model.to(StrToModel.device)
    
    ae_model.eval()
    print ('\nModel loaded.')
    
    
    # For batch/partitioned writes. 
    max_step = len(base_film_df)
    
    steps = range(last_stop+step, max_step, step)
    for w in steps:
        update = 'Step: {} - {}.'.format(str(w-step),str(w))
        print('\r{:30}'.format(update),end='')
        writeBatchMetaData(base_film_df, ae_model, start=w-step, stop=w)
    
    writeBatchMetaData(base_film_df, ae_model, start=steps[-1], stop=max_step)


def swapAllEncodings():
    
    args = hparameters.model_args
    
    ae_model = Encoder(code_size=args['code_size'], 
                       em_len=args['em_len'], 
                       encoder_size=args['encoder_size'], 
                       n_dir=args['encoder_n_dir'])
    ae_model.load_state_dict(torch.load(os.path.abspath('../model')+'./saves/encoder_statedict.pth'))
    ae_model = ae_model.to(StrToModel.device)
    
    ae_model.eval()
    print ('\nModel loaded.')
    
    swapEncodings('synopsis_encodings', ae_model)




if __name__ == '__main__':
    
    #word2vecWriteAll()
    
    #metadataWriteAll('title_id_list_d.csv')
    
    swapAllEncodings()
    
    pass