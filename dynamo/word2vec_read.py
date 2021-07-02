import boto3
import numpy as np


# Get word vector from dynamo.
def readSingleWord(word, table_name='word2vec_gnews_300d'):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    response = table.get_item(Key={'word': word})
    return np.frombuffer(response['Item']['vector'].value)


# Get multiple word vectors at once. Returns a list of arrays.
def readBatchWords(word_list, table_name='word2vec_gnews_300d'):
    
    decode = lambda b: np.frombuffer(b, dtype=np.float32)
    
    dynamodb = boto3.resource('dynamodb')
        
    # Dynamo will only accept unique requests.
    u_word_list = np.unique(word_list)
    key_list = [{'word': str(word)} for word in u_word_list]
    response = dynamodb.batch_get_item(RequestItems={table_name: {'Keys': key_list}})
    
    # We'll instead rebuild the original input via unique responses.
    response_list = response['Responses'][table_name]
    response_dict = {}
    for r in response_list:
        response_dict[r['word']] = r['vector'].value
    
    # Ignore words which are not in the word2vec language.
    true_response = []
    for word in word_list:
        try:
            true_response.append(decode(response_dict[word]))
        except:
            pass
            
    return true_response

