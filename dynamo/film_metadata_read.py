import boto3
import numpy as np



# Given a list of nIndex values, read the corresponding titles.
def readTitlesFromInds(ind_list, table_name='all_film_data'):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    title_list = []
    for ind in ind_list:
        response = table.get_item(Key={'nIndex': ind})
        title_list.append(response['Item']['nIndex'])
    
    return title_list




# Get film encoding vector from dynamo.
def readSingleFilmEncoding(table_name='all_film_data', fID=None, title=None, long=False):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    if long: ret_key = 'long_encoding'
    else: ret_key = 'short_encoding'
    
    if fID is not None:
        response = table.get_item(Key={'fID': fID})
    else:
        response = table.get_item(Key={'title': title})
        
    return np.frombuffer(response['Item'][ret_key].value)


# Get all film vectors at once. Returns a list of arrays.
def readAllEncodings(table_name='all_film_data',ind=False,long=False):
    
    decode = lambda b: np.frombuffer(b, dtype=np.float32)
    
    if long: ret_key = 'long_encoding'
    else: ret_key = 'short_encoding'
    
    if ind: attr_list = ['nIndex', ret_key]
    else: attr_list = [ret_key]
    
    client = boto3.client('dynamodb')
    paginator = client.get_paginator('scan')
    
    page_iterator = paginator.paginate(TableName=table_name,
    AttributesToGet=attr_list)
    
    if not ind:
        true_response = []
        for page in page_iterator:
            for item in page['Items']:
                true_response.append(decode(item[ret_key]['S']))
    else:
        true_response = {}
        for page in page_iterator:
            for item in page['Items']:
                true_response[int(item['nIndex']['N'])] = decode(item[ret_key]['B'])
    
    return true_response





# Given a list of nIndex values, read the corresponding titles.
def readTitlesFromInds(ind_list, table_name='all_film_data'):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    title_list = []
    for ind in ind_list:
        response = table.get_item(Key={'nIndex': ind})
        title_list.append(response['Item']['title'])
    
    return title_list


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





# Get all plot synopses at once, generally for model training.
def readAllPlots(table_name='all_film_data',long=False):
    
    if long: ret_key = 'longplot'
    else: ret_key = 'shortplot'
    
    client = boto3.client('dynamodb')
    paginator = client.get_paginator('scan')
    
    page_iterator = paginator.paginate(TableName=table_name,
    AttributesToGet=[ret_key])
    
    count = 1
    true_response = []
    for page in page_iterator:
        for item in page['Items']:
            update = 'Step: {}.'.format(str(count))
            print('\r{:30}'.format(update),end='')
            
            true_response.append(item[ret_key]['S'])
            count+=1
    
    return true_response
