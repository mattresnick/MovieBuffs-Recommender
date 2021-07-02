import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):
    def __init__(self, code_size, em_len, decoder_size, n_dir=1, dropout=0):
        super(Decoder, self).__init__()
        
        self.code_size = code_size        # Size of code layer (small representation)
        self.em_len = em_len              # Length of word embeddings
        self.decoder_size = decoder_size  # Number of hidden units in decoder.
        self.n_dir = n_dir                # Number of directions for the LSTM unit (1 or 2)
        
        # Decoder LSTM unit structure. The input is the code layer, output is word embeddings.
        self.decoder_lstm = nn.LSTM(input_size=self.code_size, 
                                    hidden_size=self.decoder_size, 
                                    dropout=dropout,
                                    bidirectional=(n_dir==2))
        

    def forward(self, X, sentence_length):
        
        # Initialize zerohidden and cell states for decoder.
        hidden_state_d = torch.zeros(self.n_dir, 1, self.decoder_size).to(device)
        cell_state_d = torch.zeros(self.n_dir, 1, self.decoder_size).to(device)
        out_d = torch.zeros(self.n_dir, self.decoder_size).to(device)
        
        # Use code layers as first input, then use previous step output as input
        # to generate prediction sentence.
        for i in range(sentence_length):
            if i==0: 
                out_d, (hidden_state_d, cell_state_d) = \
                    self.decoder_lstm(X, 
                    (hidden_state_d, 
                     cell_state_d))
                all_output = out_d
            else: 
                out_d, (hidden_state_d, cell_state_d) = \
                    self.decoder_lstm(out_d, 
                    (hidden_state_d, 
                     cell_state_d))
                
                all_output = torch.cat((all_output,out_d),0)
        
        return all_output


