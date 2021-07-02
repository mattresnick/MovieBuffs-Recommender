import os
import io
import boto3
import json
import csv

runtime= boto3.client('runtime.sagemaker')

SKLearn_Endpoint = 'moviebuffs-sklearn'
PyTorch_Endpoint = 'moviebuffs-pytorch'

# Check if film is in our database by ID.
def checkID(ID):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('synopsis_encodings')
    response = table.get_item(Key={'fID': ID})
    return 'Item' in list(response.keys())



def lambda_handler(event, context):
    # Event should be formatted like:
    # {"data":data, "type":type=[title, ID, phrase]}
    print("Received event: " + json.dumps(event, indent=2))
    
    incoming_json = json.loads(json.dumps(event))
    
    if incoming_json['type'] == 'ID':
        if checkID(incoming_json['data']):
            response = runtime.invoke_endpoint(EndpointName=SKLearn_Endpoint,
                                           Body=incoming_json)
        else:
            return json.dumps({'response_type':'failed',
                               'reason':'ID not in database'})
                
                
    elif incoming_json['type']=='title':
        response = runtime.invoke_endpoint(EndpointName=SKLearn_Endpoint,
                                       Body=incoming_json)
                                       
                                       
    elif incoming_json['type']=='phrase':
        response = runtime.invoke_endpoint(EndpointName=PyTorch_Endpoint,
                                       Body=incoming_json)
                                       
                                       
    else:
        return json.dumps({'response_type':'failed',
                           'reason':'invalid event type'})
    
                                       
    
    result = response["Body"]._raw_stream.data.decode("utf-8")
    result = json.loads(result)
    result['response_type'] = 'succeeded'
    
    return json.dumps(result)