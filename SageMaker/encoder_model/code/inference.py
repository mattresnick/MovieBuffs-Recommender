import torch
from torch import nn
import os
import numpy as np
import boto3
from six import BytesIO
import json

def toArray(val):
    if torch.cuda.is_available():
        arr = val.detach().cpu().numpy()
    else:
        arr = val.detach().numpy()
    
    return arr



def readBatchWords(word_list):
    
    decode = lambda b: np.frombuffer(b, dtype=np.float32)
    
    dynamodb = boto3.resource('dynamodb')
    table_name='word2vec_gnews_300d'
        
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
def sentenceToModelFormat(sentence):
    
    if isinstance(sentence, str):
        sentence = formatForModel(sentence)
    elif isinstance(sentence, np.ndarray):
        sentence = formatForModel(sentence[0])
    
    sentence_encodings = readBatchWords(sentence)
    input_data = torch.tensor(sentence_encodings).unsqueeze(1).float()
    
    return input_data





class Encoder(nn.Module):
    def __init__(self, code_size, em_len, encoder_size, n_dir=1, dropout=0):
        super(Encoder, self).__init__()
        
        self.code_size = code_size        # Size of code layer (small representation)
        self.em_len = em_len              # Length of word embeddings
        self.encoder_size = encoder_size  # Number of hidden units in encoder.
        self.n_dir = n_dir                # Number of directions for the LSTM unit (1 or 2)
        
        
        # Encoder LSTM unit structure. The inputs are single word embeddings.
        self.encoder_lstm = nn.LSTM(input_size=self.em_len, 
                                    hidden_size=self.encoder_size, 
                                    dropout=dropout,
                                    bidirectional=(n_dir==2))
        
        # Fully connected layer structure to obtain output at each time step.
        self.code_layer = nn.Linear(self.encoder_size, self.code_size)

    def forward(self, X):
        
        input_bs = X.size(0) # Input length in words.
        
        # Initialize random h_0 and c_0 for encoder.
        hidden_state_e = torch.zeros(self.n_dir, 1, self.encoder_size)
        cell_state_e = torch.zeros(self.n_dir, 1, self.encoder_size)
        
        # Get current encoder LSTM unit, h_n, and c_n.
        out_e, (h_e, c_e) = self.encoder_lstm(X, 
                                             (hidden_state_e, 
                                              cell_state_e))
        hidden_state_e, cell_state_e = h_e, c_e
        
        # Produce code layer from last of output.
        hidden_state_e = hidden_state_e.view(-1, self.encoder_size)[-1] #shape of this might need some retooling
        
        
        code = nn.ReLU()(self.code_layer(hidden_state_e)) # output should be (1 x code size)
        
        
        return code









def model_fn(model_dir):
    # notebook version
    model = Encoder(128, 300, 128, 2)
    
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=torch.device('cpu')))
    
    return model


def input_fn(request_body, request_content_type):
    
    if request_content_type == 'application/json':
        #incoming_decoded = BytesIO(request_body).getvalue().decode("utf-8")
        incoming_json = json.loads(request_body)
        true_data = incoming_json['data']
       
    elif isinstance(request_body, str):
        true_data = request_body
    else:
        raise Exception('input non-json: '+str(request_body)+' - '+str(request_content_type))
    
    phrase_tensor = sentenceToModelFormat(true_data)
    
    return phrase_tensor


def predict_fn(input_data, model):
    model.eval()
    out = model(input_data)
    out = toArray(out)
    return out

def output_fn(prediction, response_content_type):
    
    client = boto3.client('sagemaker-runtime')
    
    endpoint_name = 'moviebuffs-sklearn'
    
    send = json.dumps({'data':prediction.tolist(),
                       'type':'array'})
    
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=send
        )
    
    r_dict = response["Body"]._raw_stream.data.decode("utf-8")
    r_dict = json.loads(r_dict)
    res = json.dumps(r_dict)
    return res, 'application/json'


