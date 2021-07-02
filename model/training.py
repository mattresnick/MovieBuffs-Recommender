import torch
from torch import nn, optim

import copy
import numpy as np
import pandas as pd
import gensim as g
import matplotlib.pyplot as plt

from LSTMAE import LSTMAE
import hparameters as args
from model_init import readAllPlots
from StrToModel import sentenceToModelFormat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def plot_losses(tloss,vloss):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(tloss)),tloss)
    ax.plot(np.arange(len(vloss)),vloss)
    
    plt.show()





def train(model, train_data, val_data, n_epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    #criterion = nn.L1Loss(reduction='sum').to(device)
    criterion = nn.MSELoss().to(device)
    
    train_log, val_log = [], [] 
    for epoch in range(1, n_epochs+1):
        print ('\nTraining')
        
        model = model.train()
        train_losses, val_losses = [], [] 
        
        for i,plot in enumerate(train_data):
            update = 'Epoch: {}. Step: {}.'.format(str(epoch), str(i))
            print('\r{:30}'.format(update),end='')
            #print('\r{:30}'.format('Epoch: ' + str(epoch) + '. Step: ' + str(i)),end='')
            
            
            plot=plot.to(device)
            rplot = model(plot)
            
            reverse = torch.Tensor()
            for p in range(plot.size(0)-1,-1,-1):
                reverse = torch.cat((reverse,plot[p]), 0)
            reverse = reverse.view(-1,1,300)
            
            loss = criterion(rplot, reverse)
            train_losses.append(loss.item())
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            
        print ('\nValidation')
        model = model.eval()
        with torch.no_grad():
            for i,plot in enumerate(val_data):
                print('\r{:30}'.format('Epoch: ' + str(epoch) + '. Step: ' + str(i)),end='')
                
                plot=plot.to(device)
                rplot = model(plot)
                
                reverse = torch.Tensor()
                for p in range(plot.size(0)-1,-1,-1):
                    reverse = torch.cat((reverse,plot[p]), 0)
                reverse = reverse.view(-1,1,300)
                
                loss = criterion(rplot, reverse)
                val_losses.append(loss.item())
                
        train_log.append(np.mean(train_losses))
        val_log.append(np.mean(val_losses))
        
        print('\nTrain loss {}. Validation loss {}.'.format(train_log[-1], val_log[-1]))
        plot_losses(train_log, val_log)
    
        torch.save(model.encoder.state_dict(), './saves/encoder_statedict_epoch'+str(epoch)+'.pth')
        torch.save(model.state_dict(), './saves/whole_model'+str(epoch)+'.pth')
    #model.load_state_dict(best_model)
    return model, train_log, val_log








# Run a test sentence through the model to see what decoder produces.
def testSentence():
    w2v_path = 'GoogleNews-vectors-negative300.bin'
    w2v_model=g.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    
    def sentenceComparison(model, w2v_model, sentence):
        print (sentence)
        
        in_sent = sentenceToModelFormat(sentence)
        in_sent_t = torch.FloatTensor(in_sent)
        
        pred = model(in_sent_t).detach().cpu().numpy().reshape((-1,300))
        print (pred)
        
        for arr in pred:
            new_word = w2v_model.most_similar(positive=[arr], topn=1)
            print (new_word)
    
    sentenceComparison(model, w2v_model, 'the quick brown fox jumps over the lazy dog')










# Load in data which was saved to file.
def loadDataFromFile(loadpath, partial=False):
    
    # Load in raw data from file.
    arr_data = np.load(loadpath, allow_pickle=True)
    
    if not partial:
        # Convert all data to torch tensors.
        all_data = [torch.tensor(p).unsqueeze(1).float() for p in arr_data]
        return all_data
    
    else:
        # Select out a random sample (for faster testing) and convert to torch tensor.
        rand_list = np.random.randint(0,len(arr_data),5000)
        all_data = [torch.tensor(p).unsqueeze(1).float()  \
                    for i,p in enumerate(arr_data) if i in rand_list]
        return all_data


# Save data from Dynamo to file.
def loadDataToFile(savepath, long=False):
    plot_data = readAllPlots()
    
    print ('Data loaded.')
    
    # Convert to vectors via word2vec, then save to npy file.
    all_plots = []
    for i,p in enumerate(plot_data):
        update = 'Step: {}.'.format(str(i))
        print('\r{:30}'.format(update),end='')
        
        word_vecs = np.array(sentenceToModelFormat(p, False))
        all_plots.append(word_vecs)
    
    all_plots = np.array(all_plots)
    np.save(savepath, all_plots)


# Load data for training directly from DynamoDb. This is pretty slow, so in
# practice it's better to save to file and load that.
def loadDataFromDynamo(long=False):
    plot_data = readAllPlots()
    
    # Convert to vectors via word2vec, then convert to torch tensors.
    all_data = [sentenceToModelFormat(p) for p in plot_data]
    
    return all_data


#loadDataToFile(savepath='plot_vecs.npy')
all_data = loadDataFromFile(loadpath='plot_vecs.npy')


# Split data to train and validation sets.
split_int = int(len(all_data)*.8)
train_data = all_data[:split_int]
validation_data = all_data[split_int:]



# Instantiate model and train.
model = LSTMAE(**args.model_args).to(device)


model, train_log, val_log = train(model, 
                                  train_data, 
                                  validation_data, 
                                  args.num_epochs,
                                  args.learning_rate)



'''
torch.save(model.state_dict(), './saves/whole_model.pth')
with open('./saves/whole_model.pth', 'rb') as f:
    model.load_state_dict(torch.load(f))
model.to(device)
'''




