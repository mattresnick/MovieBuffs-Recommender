import boto3

# Write words to dynamo one by one.
def writeIndWords(word_model, start=0, stop=0, table_name='word2vec_gnews_300d'):
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    for i,item in enumerate(list(word_model.wv.vocab)[start:stop]):
        update = 'Step: {}.'.format(str(i))
        print('\r{:30}'.format(update),end='')
        table.put_item(Item={'word':item, 'vector':word_model[item].tostring()})
    

# Write words to dynamo in batches of 10K.
def writeBatchWords(word_model, start=0, stop=0, table_name='word2vec_gnews_300d'):
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    with table.batch_writer() as batch:
        for i,item in enumerate(list(word_model.wv.vocab)[start:stop]):
            batch.put_item(Item={'word':item, 'vector':word_model[item].tostring()})

