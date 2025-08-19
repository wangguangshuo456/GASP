from re import X
import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset  
import numpy as np
import torch.nn.functional as F
import math

import torch.nn.utils.rnn as rnn_utils




class TrData(Dataset): 
    def __init__(self, data_seq):
        self.data_seq = data_seq

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return self.data_seq[idx]


def collate_fn(trs): 
    onetr = trs[0] 
    onepoint_size = onetr.size(1) 
    
    
    
    input_size = onepoint_size -1
    
    
    
    
    trs.sort(key=lambda x: len(x), reverse=True)
    
    tr_lengths = [len(sq) for sq in trs]


    
    trs = rnn_utils.pad_sequence(trs, batch_first=True, padding_value=0)
   

    

    
    var_x = trs[:,:,1:input_size+1] 
    
    
    
    tmpy = trs[:,:,0]
    
   
    
    
    var_y = tmpy[:,0] 
    
    
    return var_x, var_y, tr_lengths  

class lstm(nn.Module):
    def __init__(self,input_size=2,hidden_size=4,num_layers=1,num_classes=2,batch_size=1):
        super(lstm,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.layer1 = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.layer2 = nn.Linear(hidden_size,num_classes)
    
    def forward(self,x,len_of_oneTr):
        
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda() 
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        

        
        batch_x_pack = rnn_utils.pack_padded_sequence(x,
                                                  len_of_oneTr, batch_first=True).cuda()

        

        
        out, (h1,c1) = self.layer1(batch_x_pack, (h0, c0))  
        

        
        out = self.layer2(h1)  
        return out, h1  
    


class LSTM_Attention(nn.Module):
    def __init__(self,input_size=2,hidden_size=4,num_layers=1,num_classes=2,batch_size=1):
        super(LSTM_Attention,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.layer1 = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        
        self.layer3 = nn.Linear(hidden_size*2,num_classes)
        
        self.relu = nn.ReLU()


    def forward(self,x,len_of_oneTr):
        
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda() 
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        

        
        batch_x_pack = rnn_utils.pack_padded_sequence(x,
                                                  len_of_oneTr, batch_first=True).cuda()

        

        
        out, (h1,c1) = self.layer1(batch_x_pack, (h0, c0))   


        
        outputs , lengths = rnn_utils.pad_packed_sequence(out, batch_first=True)
        
        
        permute_outputs = outputs.permute(1,0,2)  
       

        atten_energies = torch.sum(h1*permute_outputs, dim=2)  
        

        atten_energies = atten_energies.t() 

        scores = F.softmax(atten_energies, dim=1) 

        scores = scores.unsqueeze(0)
        

        permute_permute_outputs = permute_outputs.permute(2,1,0)
        
        context_vector = torch.sum(scores*permute_permute_outputs,dim=2)  

        
        context_vector = context_vector.t()
        context_vector = context_vector.unsqueeze(0)
        
        out2 = torch.cat((h1,context_vector),2)
       
        
        out = self.layer3(out2)

        return out, out2  



