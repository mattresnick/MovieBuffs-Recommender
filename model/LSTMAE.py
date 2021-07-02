import torch
from torch import nn
from Encoder import Encoder
from Decoder import Decoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LSTMAE(nn.Module):
    def __init__(self, em_len, code_size, 
                 encoder_size, encoder_n_dir, encoder_dropout,
                 decoder_size, decoder_n_dir, decoder_dropout):
        
        super(LSTMAE, self).__init__()
        
        self.encoder = Encoder(code_size, 
                               em_len, 
                               encoder_size, 
                               encoder_n_dir, 
                               encoder_dropout).to(device)
        
        self.decoder = Decoder(code_size, 
                               em_len, 
                               decoder_size, 
                               decoder_n_dir, 
                               decoder_dropout).to(device)
        
        
    def forward(self, X):
        code = self.encoder(X)
        out = self.decoder(code,X.size(0))
        return out
