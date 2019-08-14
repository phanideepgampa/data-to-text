import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
from torch.nn import Parameter
from copy import deepcopy
import numpy as np


class Bandit(nn.Module):
    def __init__(self,config):
        super(Bandit,self).__init__()
        
        # Parameters
        self.drop = config.dropout
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.LSTM_layers = config.LSTM_layers
        self.LSTM_hidden_units = config.LSTM_hidden_units
        self.train_embed = config.train_embed

        #Layers
        self.dropout = nn.Dropout(self.drop)
        self.word_embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        # self.word_embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embedding))
        if self.train_embed == False:
            self.word_embedding.weight.requires_grad=False

        self.record_encoder = nn.Sequential(nn.Linear(4*self.embedding_dim,self.embedding_dim),
                                            nn.ReLU(),
                                            self.dropout)
        self.attention_matrices = nn.ParameterList([nn.Parameter(torch.randn(self.embedding_dim,self.embedding_dim)),nn.Parameter(torch.randn(2*self.embedding_dim,self.embedding_dim))])
        self.lstm_layer = nn.LSTM(input_size=self.embedding_dim,
                                 hidden_size=self.LSTM_hidden_units,
                                 batch_first=True,
                                 num_layers=1,
                                 dropout= self.drop)
        self.probability_layer = nn.Sequential(nn.Linear(self.embedding_dim,self.embedding_dim),
                                               nn.Tanh(),
                                               self.dropout ) 
        self.normalisation_matrix = nn.Parameter(torch.randn(self.embedding_dim,1))
        
    def content_selection(self,records):
        att_matrix = torch.matmul(torch.matmul(records,self.attention_matrices[0]),records.transpose(1,0)) # (r,r)
        c_i= torch.matmul(F.softmax(att_matrix,-1),records) #(r,emb_dim)
        r_att = torch.matmul(torch.cat([records,c_i],dim=-1),self.attention_matrices[1]) #(r,emb_dim)
        r_cs = records*(F.sigmoid(r_att))
        return r_cs 

    def forward(self,records): #input context tokens      
        records = self.word_embedding(records)  # shape-(r,4,emb_dim)
        encoded_records = self.record_encoder(records.view(-1,4*self.embedding_dim)) # shape (r,emb_dim)
        r_cs = self.content_selection(encoded_records) #(r,emb_dim)
        r_cs_mean = torch.mean(r_cs,0) #(emb_dim)
        lstm_output,_ = self.lstm_layer(r_cs.unsqueeze(0),(r_cs_mean.view(1,1,-1),r_cs_mean.view(1,1,-1))) #(1,r,emb_dim)
        lstm_output = lstm_output.view(-1,self.LSTM_hidden_units)
        prob_out = self.probability_layer(lstm_output) 
        prob = F.softmax(torch.matmul(prob_out,self.normalisation_matrix),0) #(r,1)
        return prob.view(-1,1),lstm_output


class Generator(nn.Module):
    def __init__(self,config):
        super(Generator,self).__init__()
        # Parameters
        self.drop = config.dropout
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.LSTM_layers = config.LSTM_layers
        self.LSTM_hidden_units = config.LSTM_hidden_units
        self.decode_type = config.decode_type
        self.train_embed = config.train_embed

        if self.decode_type == 'joint':
            self.decode_criterion =  self.joint_copy
        elif self.decode_type == 'conditional':
            self.decode_criterion =  self.conditional_copy
            self.conditional_switch = nn.Sequential(nn.Linear(self.LSTM_hidden_units,1),
                                                    nn.Sigmoid() )


        #layers
        self.dropout = nn.Dropout(self.drop)
        self.word_embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        if self.train_embed == False:
            self.word_embedding.weight.requires_grad=False
        self.first_lstm = nn.LSTM(input_size=self.embedding_dim,
                                 hidden_size=self.LSTM_hidden_units,
                                 batch_first=True,
                                 num_layers=self.LSTM_layers,
                                 bidirectional = True,
                                 dropout= self.drop)
        self.decoder = nn.LSTM(input_size=self.embedding_dim,
                                 hidden_size=self.LSTM_hidden_units,
                                 batch_first=True,
                                 num_layers=self.LSTM_layers,
                                 dropout= self.drop,
                                 bidirectional=True)
        self.attention_matrices = nn.ParameterList([nn.Parameter(torch.randn(self.LSTM_hidden_units,2*self.LSTM_hidden_units)),
                                                    nn.Parameter(torch.randn(3*self.LSTM_hidden_units,self.LSTM_hidden_units))])
        self.outputlayer = nn.Sequential(nn.Linear(self.LSTM_hidden_units,self.vocab_size),
                                        nn.Softmax(-1))
    
    def joint_copy(self,beta_t,p_gen,d_t,z_k):
        p_out = p_gen.clone()
        for i,z in enumerate(z_k):
            p_out[:,z]+=beta_t[:,i]
        return p_out/torch.sum(p_out,-1)
    def conditional_copy(self,beta_t,p_gen,d_t,z_k):
        p_switch = self.conditional_switch(d_t)
        p_out = p_gen.clone()*(1-p_switch)
        beta = beta_t.clone()*p_switch
        for i,z in enumerate(z_k):
            p_out[:,z]+=beta_t[:,i]
        return p_out/torch.sum(p_out,-1)

    def forward_step(self,prev_emb,prev_hidden,content_plan,e_k,z_k):
        prev_emb = prev_emb.view(1,1,self.embedding_dim)
        output,hidden = self.decoder(prev_emb,prev_hidden)
        d_t = torch.mean(hidden[0],0)
        e_k = e_k.squeeze(0)
        beta_t = torch.matmul(torch.matmul(d_t.view(-1,self.LSTM_hidden_units),self.attention_matrices[0]),e_k.transpose(1,0)) #(1,z)
        q_t = torch.matmul(F.softmax(beta_t,-1),e_k)
        d_att = F.tanh(torch.matmul(torch.cat([d_t,q_t],-1),self.attention_matrices[1])) #(1,lstm_hidden/emb_dim)
        d_att = self.dropout(d_att)
        prob_y = self.outputlayer(d_att)
        prob_out = self.decode_criterion(beta_t,prob_y,d_t,z_k)
        return prob_out, hidden

    def forward(self,content_plan,vocab):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lstm_output, hidden = self.first_lstm(content_plan.view(1,-1,self.embedding_dim)) 
        return lstm_output, hidden , self.word_embedding(torch.LongTensor([vocab['<sos>']]).to(device))
    def get_embedding(self,index):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.word_embedding(torch.LongTensor([index]).to(device))



        
