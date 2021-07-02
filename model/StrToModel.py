import torch
import numpy as np
import boto3

from model_init import readBatchWords


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
from __init__ import readSingleWord
import gensim as g
w2v_path = 'GoogleNews-vectors-negative300.bin'
w2v_model=g.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
'''




def formatForModel(f_str):
    '''
    Parameters:
        - f_str (string): String of raw plot sentence(s).

    Returns:
        - list of strings, no punctuation, no uppercase.

    '''
    
    f_str = f_str.translate(str.maketrans('', '', r"""!"#$%&'()*+,-.:;=?@[\]^_`{|}~""")).lower()
    f_str = f_str.replace(u'\u201C', '')
    f_str = f_str.replace(u'\u201D', '')
    f_str = f_str.replace(u'\u2018', '')
    f_str = f_str.replace(u'\u2019', '')
    f_str = f_str.replace('\'', '')
    
    
    f_str = list(filter(None, f_str.split())) 
    
    return f_str



# Convert English sentence into encoder model input format.
def sentenceToModelFormat(sentence,tensor=True):
    
    if isinstance(sentence, str):
        sentence = formatForModel(sentence)
    
    sentence_encodings = readBatchWords(sentence)
    
    if tensor:
        input_data = torch.tensor(sentence_encodings).unsqueeze(1).float().to(device)
    else:
        input_data = sentence_encodings
    
    return input_data




# Depricated functions now that we can make calls to Dynamo.
'''
# Df vector to string.
def vecToStr(row):
    enc = row['encoding']
    
    new_str=''
    for val in enc:
        new_str = new_str + str(val) + ','
    
    return new_str[:-1]


# Use my custom delimeters to splice out data.
def strToArr(in_str):
    word_strs = in_str.split('|')
    
    word_vecs = []
    for vec in word_strs:
        word_vecs.append(np.array([float(part) for part in vec.split(',')]))
    
    return np.array(word_vecs)


# Get string-casted vector representations for each word.
def toVecRepStr(plot, cent=False):
    global w2v_model
    
    if w2v_model is None:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('word2vec_gnews_300d')
    
    plot_list = formatForModel(plot)
    
    vecs = ''
    for word in plot_list:
        try: 
            
            if w2v_model is None:
                vec_word = readSingleWord(word,dynamodb,table)
            
            else:
                vec_word = w2v_model[word]
            
            add_word = ''
            for val in vec_word:
                add_word = add_word+str(val)+','
            
            
            vecs = vecs+str(add_word[:-1])+'|'
        
        # Skip over words not found in dictionary.
        except: pass 
    
    return vecs[:-1]



# Convert English sentence into encoder model input format, via string 
# conversion for save. Depricated.
def sentenceToModelFormatSave(sentence):
    vec_sentence = toVecRepStr(sentence)
    sentence_input = strToArr(vec_sentence)
    input_data = torch.tensor(sentence_input).unsqueeze(1).float().to(device)
    
    return input_data
'''