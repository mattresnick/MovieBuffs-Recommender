import os
import numpy as np
import boto3
import pickle
import requests
from six import BytesIO
import json

def OMDbRequest(key, ID=None, title=None, full_plot=False):
    
    if ID is not None:
        r_str = 'http://www.omdbapi.com/?apikey={}&i={}'.format(key,ID)
    else:
        if not isinstance(title, str):
            title = str(title)
        rtitle = title.replace(' ','+')
        r_str = 'http://www.omdbapi.com/?apikey={}&t={}'.format(key,rtitle)
    
    if full_plot: r_str = r_str+'&plot=full'
    
    r = requests.get(r_str)
    
    return r.json()




def OMDbSearchReturnBest(key, title=None):
    
    rtitle = title.replace(' ','+')
    r_str = 'http://www.omdbapi.com/?apikey={}&s={}&type=movie'.format(key,rtitle)
    
    
    r = requests.get(r_str)
    r_json = r.json()
    
    if r_json['Response']=='False': return
    
    sort_list = [[i['imdbID'], i['imdbVotes']] for i in r_json['Search']]
    sort_list.sort(reverse=True, key=lambda x: x[1])
    
    if len(sort_list)>0: 
        return sort_list[0][0]
    
    return




# Given a list of nIndex values, read the corresponding titles.
def readTitlesFromInds(ind_list, table_name='all_film_data'):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    title_list, id_list = [], []
    for ind in ind_list:
        response = table.get_item(Key={'nIndex': ind})
        title_list.append(response['Item']['title'])
        id_list.append(response['Item']['fID'])
    
    return title_list, id_list


# Given an fID, return encoding.
def readEncodingFromID(fID, table_name='synopsis_encodings',long=False):
    decode = lambda b: np.frombuffer(b, dtype=np.float32)
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    if long: ret_key = 'long_encoding'
    else: ret_key = 'short_encoding'
    
    response = table.get_item(Key={'fID': fID})
    r_encoding = decode(response['Item'][ret_key].value)
    
    return r_encoding




# Get recommendations, given a film title or sequence encoding.
def nearestTitles(model, title=None, fID=None, encoding=None):
    if encoding is None:
        
        # Search for fID based on title.
        if fID is None:
            r_json = OMDbRequest(key='', title=title)

            if r_json['Response']=='False':
                fID = OMDbSearchReturnBest(key='', title=title)
            else:
                fID = r_json['imdbID']
        
        encoding = readEncodingFromID(fID)
    
    
    neighbor = model.kneighbors([encoding])[1]
    ind_list = [int(n) for n in neighbor[0]]
    
    titles, IDs = readTitlesFromInds(ind_list)
    
    return titles, IDs









def model_fn(model_dir):
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model = pickle.load(f)
    
    return model


def input_fn(request_body, request_content_type):
    
    if request_content_type == 'application/json':
        #incoming_decoded = BytesIO(request_body).getvalue().decode("utf-8")
        incoming_json = json.loads(request_body)
        
        if incoming_json['type']=='array':
            true_data = [np.array(incoming_json['data']),'array']
            
        elif incoming_json['type']=='title':
            true_data = [incoming_json['data'],'title']
            
        elif incoming_json['type']=='ID':
            true_data = [incoming_json['data'],'ID']
            
        else:
            raise Exception('input json: '+str(request_body)+' - '+str(request_content_type))
        
        
       
    elif isinstance(request_body, str):
        return request_body
    else:
        raise Exception('input non-json: '+str(request_body)+' - '+str(request_content_type))
       
    return true_data

def predict_fn(input_data, model):
    
    if isinstance(input_data, str):
        titles, IDs = nearestTitles(model,title=input_data)
    elif isinstance(input_data, list):
        if input_data[1]=='title':
            titles, IDs = nearestTitles(model,title=input_data[0])
        elif input_data[1]=='ID':
            titles, IDs = nearestTitles(model,fID=input_data[0])
        elif input_data[1]=='array':
            titles, IDs = nearestTitles(model,encoding=input_data[0])
        else:
            raise Exception('predict list: '+str(input_data)+' - '+str(type(input_data)))
    else:
        raise Exception('predict type: '+str(input_data)+' - '+str(type(input_data)))
    
    return [titles, IDs]

def output_fn(prediction, response_content_type):
    
    res = json.dumps({'titles':prediction[0],'IDs':prediction[1]})
    
    return res, 'application/json'




