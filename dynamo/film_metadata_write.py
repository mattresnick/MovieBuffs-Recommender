import boto3
import numpy as np


from dynamo_init import StrToModel, key, tkey
from dynamo_init import omdb_retrieval as o
from dynamo_init import tmdb_retrieval as t

# Retrieves metadata for a given film.
def getMetaData(row):
    
    fID=row['fID']
    
    req = o.OMDbRequest(key,ID=fID,full_plot=True)
    longplot = req['Plot']
    genre = req['Genre']
    rating = req['imdbRating']
    votenum = req['imdbVotes']
    
    tfID = t.IMDBtoTMDB(fID, tkey)
    
    cast = np.array([]).tostring()
    crew_roles, crew_names = np.array([]).tostring(), np.array([]).tostring()
    
    if tfID is not None:
        cast,crew = t.getCastandCrew(tfID, tkey)
        
        if cast is not None:
            cast = np.array(cast).tostring()
        
        if crew is not None:
            crew_roles, crew_names = np.array(crew.keys()).tostring(), np.array(crew.values()).tostring()
        
    
    return [longplot, cast, crew_roles, crew_names, genre, rating, votenum]
    

# Converts short and long plot synopses to vector encodings.
def plotToEnc(row, meta, model):
    
    short_plot, long_plot = row['shortplot'], meta['longplot']
    
    short_f = StrToModel.sentenceToModelFormat(short_plot)
    long_f = StrToModel.sentenceToModelFormat(long_plot)
    
    short_enc = model(short_f)
    short_enc = short_enc.detach().cpu().numpy()
    
    long_enc = model(long_f)
    long_enc = long_enc.detach().cpu().numpy()
    
    return [short_enc.tostring(), long_enc.tostring()]





def writeBatchMetaData(movie_df, model, start=0, stop=0, table_name='all_film_data'):
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    write_df = movie_df.iloc[start:stop]
    
    with table.batch_writer() as batch:
        for i,item in write_df.iterrows():
            update = 'Entry: {}.'.format(str(i))
            print('\r{:30}'.format(update),end='')
            
            metadata_list = ['longplot', 'cast', 'crew_roles', 
                             'crew_names', 'genre', 'rating', 'votenum']
            meta = {metadata_list[i]:m for i,m in enumerate(getMetaData(item))}
            
            enc_list = ['short_encoding', 'long_encoding']
            enc = {enc_list[j]:e for j,e in enumerate(plotToEnc(item, meta, model))}
            
            batch.put_item(Item={'fID':item['fID'], 
                                 'title':item['title'], 
                                 'short_encoding':enc['short_encoding'], 
                                 'long_encoding':enc['long_encoding'], 
                                 'shortplot':item['shortplot'], 
                                 'longplot':meta['longplot'], 
                                 'cast':meta['cast'], 
                                 'crew_roles':meta['crew_roles'], 
                                 'crew_names':meta['crew_names'], 
                                 'genre':meta['genre'], 
                                 'rating':meta['rating'], 
                                 'votenum':meta['votenum'],
                                 'nIndex':i
                                 })




# Add nIndex column for use with Nearest Neighbor component.
def addIndex(table_name):
    
    client = boto3.client('dynamodb')
    paginator = client.get_paginator('scan')
    
    page_iterator = paginator.paginate(TableName=table_name,
    AttributesToGet=['fID'])
    
    ID_list = []
    for page in page_iterator:
        for item in page['Items']:
            ID_list.append(item['fID']['S'])
    
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    
    for i,fID in enumerate(ID_list):
        response = table.update_item(
                    Key={'fID':fID},
                    UpdateExpression="set nIndex=:i",
                    ExpressionAttributeValues={
                        ':i': i},
                    ReturnValues="UPDATED_NEW")
    
    return response



# Replace existing encodings with new encodings (after model update).
# Currently only updates shortplot.
def swapEncodings(table_name, model):
    
    client = boto3.client('dynamodb')
    paginator = client.get_paginator('scan')
    
    page_iterator = paginator.paginate(TableName=table_name,
    AttributesToGet=['fID','shortplot'])
    
    c=0
    ID_list = []
    for page in page_iterator:
        for item in page['Items']:
            
            update = 'Read step: {}.'.format(str(c))
            print('\r{:30}'.format(update),end='')
            c+=1
            
            short_plot = item['shortplot']['S']
            short_f = StrToModel.sentenceToModelFormat(short_plot)
            short_enc = model(short_f)
            short_enc = short_enc.detach().cpu().numpy().ravel()
             
            ID_list.append([item['fID']['S'],short_enc.tostring()])
    
    c=0
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    
    for id_enc in ID_list:
        update = 'Update step: {}.'.format(str(c))
        print('\r{:30}'.format(update),end='')
        c+=1
        
        response = table.update_item(
                    Key={'fID':id_enc[0]},
                    UpdateExpression="set short_encoding=:i",
                    ExpressionAttributeValues={
                        ':i': id_enc[1]},
                    ReturnValues="UPDATED_NEW")
    
    return response






# Port to new table, since the partition key cannot be changed.
def writeNewTable(old_table_name='synopsis_encodings', new_table_name='all_film_data'):
    
    client = boto3.client('dynamodb')
    paginator = client.get_paginator('scan')
    
    page_iterator = paginator.paginate(TableName=old_table_name)
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(new_table_name)
    
    i=0
    for page in page_iterator:
        with table.batch_writer() as batch:
            for item in page['Items']:
                update = 'Entry: {}.'.format(str(i))
                print('\r{:30}'.format(update),end='')
                i+=1
                
                item_add = {'nIndex':int(item['nIndex']['N'])}
                item_add.update({k:list(v.values())[0] for k,v in item.items() if k!='nIndex'})
                #print (item_add)
                batch.put_item(Item=item_add)



