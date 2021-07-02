import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



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
        
        # Optional tranform for code layer.
        self.code_layer = nn.Linear(self.encoder_size*self.n_dir//2, self.code_size)

    def forward(self, X):
        # input_bs = X.size(0) # Input length in words.
        
        # Initialize zero hidden, cell state for encoder.
        hidden_state_e = torch.zeros(self.n_dir, 1, self.encoder_size).to(device)
        cell_state_e = torch.zeros(self.n_dir, 1, self.encoder_size).to(device)
        
        # Get current encoder LSTM unit, h_n, and c_n.
        out_e, (hidden_state_e, cell_state_e) = \
            self.encoder_lstm(X, 
            (hidden_state_e, 
             cell_state_e))
        
        hidden_state_e = hidden_state_e.view(1,1, self.encoder_size*self.n_dir)
        #code = nn.Tanh()(self.code_layer(hidden_state_e)) #(1 x code size)
        
        return hidden_state_e
        
        

