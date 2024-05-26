import numpy as np
import pandas as pd
import torch
from torch import nn
import re
#from sklearn.utils import shuffle
from utils import metricsCal
from torch.utils.data import DataLoader,TensorDataset
from sklearn.svm import SVC
import math
import sys
import copy
import pickle
#from torch.autograd import Variable
from sklearn.model_selection import KFold
import torch.nn.functional as F
import os

#this is attention module.
def attention(query, key, value, mask=None, dropout=None):  # q,k,v: [batch, h, seq_len, d_k]
    d_k = query.size(-1)  # dim of query
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  #[batch, h, seq_len, seq_len]
    p_atten = F.softmax(scores, dim=-1)  #[batch, h, seq_len, seq_len]
    if dropout is not None:
        p_atten = dropout(p_atten)
    return torch.matmul(p_atten, value), p_atten

#this is position encoding scheme.
class PositionalEncoding(nn.Module):

    def __init__(self, dim1, dim2, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        #if dim % 2 != 0:
        #    raise ValueError("Cannot use sin/cos positional encoding with "
        #                     "odd dim (got dim={:d})".format(dim))

        """
        PE(pos,2i/2i+1) = sin/cos(pos/10000^{2i/d_{model}})
        """
        pe = torch.zeros(max_len, dim2)  #
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term0 = torch.exp((torch.arange(0, dim2, 2, dtype=torch.float) * -(math.log(10000.0) / dim2)))
        div_term1 = torch.exp((torch.arange(1, dim2, 2, dtype=torch.float) * -(math.log(10000.0) / dim2)))
        
        pe[:, 0::2] = torch.sin(position.float() * div_term0)
        pe[:, 1::2] = torch.cos(position.float() * div_term1)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        #self.drop_out = nn.Dropout(p=dropout)
        self.dim2 = dim2
        self.bm1 = nn.BatchNorm1d(dim1,eps=1e-05)

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim2)
        if step is None:
            #emb = torch.tensor(emb) + self.pe[:,:emb.shape[1]]
            emb = emb.clone().detach().requires_grad_(True) + self.pe[:,:emb.shape[1]]
        else:
            emb = emb + self.pe[step]
        #emb = self.drop_out(emb)
        emb = self.bm1(emb.to(torch.float32))
        return emb

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module)
                          for _ in range(N)])

#this is self-attention module.
class SelfAttention(nn.Module):

    def __init__(self,embedding_dim, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,query,key,value,mask=None):  # q,k,v: [batch, seq_len, embedding_dim]
        nbatches = query.shape[0]
        query, key, value = [
            l(x) for l, x in zip(self.linears,
                            (query.to(torch.float32),
                             key.to(torch.float32),
                             value.to(torch.float32) ) )
        ]
        attn, p_atten = attention(query,key,value,mask=mask,dropout=self.dropout)
        out = self.linears[-1](attn)
        return out,p_atten
# This is MultiSelf-Attention Module.
class MultiSelfAttention(nn.Module):

    def __init__(self, h,embedding_dim ,dropout=0.1):
        super(MultiSelfAttention, self).__init__()
        self.h = h
        self.attn_modules = clones(SelfAttention(embedding_dim), h)

    def forward(self,query,key,value,mask=None):  # q,k,v: [batch, seq_len, embedding_dim]
        for i in range(self.h):
            
            if i != 0:
                out_one,attn_one= self.attn_modules[i](query,key,value)
                # out = torch.add(out,torch.tensor(out_one))
                # attn = torch.add(attn,torch.tensor(attn_one))
                out = torch.add(out,out_one.clone().detach().requires_grad_(True))
                attn = torch.add(attn,attn_one.clone().detach().requires_grad_(True))
            else:
                out,attn = self.attn_modules[i](query,key,value)

        return out,attn

class MultiHeadAttention(nn.Module):

    def __init__(self, h, embedding_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        #assert embedding_dim % h == 0 
        self.d_k = embedding_dim // h  #
        self.h = h  
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)  #    
        self.dropout = nn.Dropout(p=dropout)  #

    def forward(self,query,key,value,mask=None):  # q,k,v: [batch, seq_len, embedding_dim]
        #if mask is not None:
        #    mask = mask.unsqueeze(1)  # [batch, seq_len, 1]
        nbatches = query.shape[0]  #

        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  #
            for l, x in zip(self.linears,
                            (query.to(torch.float32),
                             key.to(torch.float32), 
                             value.to(torch.float32) ) )
        ]  #
        attn, p_atten = attention(query,key,value,mask=mask,dropout=self.dropout)
        # 3. "Concat" using a view and apply a final linear.
        # [batch, h, seq_len, d_k]->[batch, seq_len, embedding_dim]
        attn = attn.transpose(1,2).contiguous().view(nbatches, -1,self.h * self.d_k)
        out = self.linears[-1](attn)
        return out,attn
    
class BahdanauAttention(nn.Module):
    """
    input: from RNN module h_1, ... , h_n (batch_size, seq_len, units*num_directions),
                                    h_n: (num_directions, batch_size, units)
    return: (batch_size, num_task, units)
    """
    def __init__(self,in_features, hidden_units,num_task):
        super(BahdanauAttention,self).__init__()
        self.W1 = nn.Linear(in_features=in_features,out_features=hidden_units)
        self.W2 = nn.Linear(in_features=in_features,out_features=hidden_units)
        self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

    def forward(self, hidden_states, values):
        #hidden_with_time_axis = torch.unsqueeze(hidden_states,dim=1)
        score  = self.V(nn.Tanh()(self.W1(values)+self.W2(hidden_states)))
        attention_weights = nn.Softmax(dim=1)(score)
        #print(attention_weights.shape)
        values = torch.transpose(values,1,2)   # transpose to make it suitable for matrix multiplication
        #print(attention_weights.shape,values.shape)
        context_vector = torch.matmul(values,attention_weights)
        context_vector = torch.transpose(context_vector,1,2)
        return context_vector, attention_weights
    
#BiLSTM+Self-Attention
class ModelBS(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.1):
        super(ModelBS, self).__init__()
        self.posi = PositionalEncoding(dim1,dim2,dropout)
        self.self_A = SelfAttention(dim2)
        self.self_B = SelfAttention(dim1)
        self.lstm = nn.LSTM(input_size=dim2,hidden_size=dim2,batch_first=True,bidirectional=False)
        self.conv1 = nn.Conv1d(dim1,dim1,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(dim1,dim1,kernel_size=3,padding=1)
        self.bm1 = nn.BatchNorm1d(dim1)

    def forward(self, x):
        
        x1 = self.conv1(x)
        x1 = self.conv2(x1)  
        x2 = self.posi(x)
        
        x2,(h_n,c_n) = self.lstm(x2)
        if x2.shape[1] == 1:
            x2 = x2.view(x2.shape[0],x2.shape[2],x2.shape[1])
            x2,attn = self.self_B(x2,x2,x2)
        else:
            x2,attn = self.self_A(x2,x2,x2)
            
        if x2.shape[2]==1:
            x2 = x2.view(x2.shape[0],x2.shape[2],x2.shape[1])
        out = x1+x2
        out = self.bm1(out)
        #out = out.view(out.shape[0],-1)
        return out,x2

class ModelBS_Pro(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.1):
        super(ModelBS_Pro, self).__init__()
        self.BS1 = ModelBS(dim1,dim2) 
        self.BS2 = ModelBS(dim1,dim2)
        self.conv1 = nn.Conv1d(dim1,dim1,kernel_size=3,padding=1)
        self.fn1 = nn.Linear(dim1*dim2,128)
        self.fn2 = nn.Linear(128,1)
        self.fn3 = nn.Linear(dim1*dim2,128)
        self.fn4 = nn.Linear(128,1)
        self.ac = nn.Sigmoid()
        self.bm1 = nn.BatchNorm1d(dim1)

    def forward(self, x):
        x1,x2 = self.BS1(x)
        x1,x2 = self.BS2(x1)
        x1 = self.bm1(x1)
        x1 = self.conv1(x1)
        x1 = x1.contiguous().view(x1.shape[0],-1)
        x1 = self.fn1(x1)
        x1 = self.fn2(x1)
        
        x2 = x2.contiguous().view(x2.shape[0],-1)
        x2 = self.fn3(x2)
        x2 = self.fn4(x2)
        out = x1+x2      
        out = self.ac(out)
        return out

#BiLSTM+MultiHead-Attention
class ModelB_Multi(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.1):
        super(ModelB_Multi, self).__init__()
        self.posi = PositionalEncoding(dim1,dim2,dropout)
        self.self_A = MultiHeadAttention(10,dim2)
        self.self_B = MultiHeadAttention(10,dim1)
        self.lstm = nn.LSTM(input_size=dim2,hidden_size=dim2,batch_first=True,bidirectional=False)
        self.conv1 = nn.Conv1d(dim1,dim1,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(dim1,dim1,kernel_size=3,padding=1)
        self.bm1 = nn.BatchNorm1d(dim1)

    def forward(self, x):
        
        x1 = self.conv1(x)
        x1 = self.conv2(x1)  
        x2 = self.posi(x)
        x2,(h_n,c_n) = self.lstm(x2)
        if x2.shape[1] == 1:
            x2 = x2.view(x2.shape[0],x2.shape[2],x2.shape[1])
            x2,attn = self.self_B(x2,x2,x2)
        else:
            x2,attn = self.self_A(x2,x2,x2)
        if x2.shape[2]==1:
            x2 = x2.view(x2.shape[0],x2.shape[2],x2.shape[1])
        out = x1+x2
        out = self.bm1(out)
        #out = out.view(out.shape[0],-1)
        return out,x2
    
class ModelB_Multi_Pro(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.1):
        super(ModelB_Multi_Pro, self).__init__()
        self.BS1 = ModelB_Multi(dim1,dim2)
        self.BS2 = ModelB_Multi(dim1,dim2)

        self.conv1 = nn.Conv1d(dim1,dim1,kernel_size=3,padding=1)
        self.fn1 = nn.Linear(dim1*dim2,128)
        self.fn2 = nn.Linear(128,1)
        self.fn3 = nn.Linear(dim1*dim2,128)
        self.fn4 = nn.Linear(128,1)
        self.ac = nn.Sigmoid()
        self.bm1 = nn.BatchNorm1d(dim1)

    def forward(self, x):
        x1,x2 = self.BS1(x)
        x1,x2 = self.BS2(x1)

        x1 = self.bm1(x1)
        x1 = self.conv1(x1)
        x1 = x1.contiguous().view(x1.shape[0],-1)
        x1 = self.fn1(x1)
        x1 = self.fn2(x1)
        
        x2 = x2.contiguous().view(x2.shape[0],-1)
        x2 = self.fn3(x2)
        x2 = self.fn4(x2)
        
        out = x1+x2
        
        out = self.ac(out)
        return out
    
#BiLSTM+Bah-Attention
class ModelB_Bah(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.1):
        super(ModelB_Bah, self).__init__()
        self.posi = PositionalEncoding(dim1,dim2,dropout)
        self.self_A = BahdanauAttention(dim2,dim2,dim1)
        self.self_B = BahdanauAttention(dim1,dim1,dim2)
        self.lstm_A = nn.LSTM(input_size=dim2,hidden_size=dim2,batch_first=True,bidirectional=False)
        self.lstm_B = nn.LSTM(input_size=dim1,hidden_size=dim1,batch_first=True,bidirectional=False)
        self.conv1 = nn.Conv1d(dim1,dim1,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(dim1,dim1,kernel_size=3,padding=1)
        self.bm1 = nn.BatchNorm1d(dim1)

    def forward(self, x):
        
        x1 = self.conv1(x)
        x1 = self.conv2(x1)  
        x2 = self.posi(x)
        
        if x2.shape[1] == 1:
            x2 = x2.view(x2.shape[0],x2.shape[2],x2.shape[1])
            x2,(h_n,c_n) = self.lstm_B(x2)
            h_n = h_n.view(h_n.shape[1],h_n.shape[0],h_n.shape[2])
            x2,attn = self.self_B(h_n,x2)
        else:
            x2,(h_n,c_n) = self.lstm_A(x2)
            h_n = h_n.view(h_n.shape[1],h_n.shape[0],h_n.shape[2])
            x2,attn = self.self_A(h_n,x2)
        if x2.shape[2]==1:
            x2 = x2.view(x2.shape[0],x2.shape[2],x2.shape[1])
        out = x1+x2
        out = self.bm1(out)
        #out = out.view(out.shape[0],-1)
        return out,x2
    
class ModelB_Bah_Pro(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.1):
        super(ModelB_Bah_Pro, self).__init__()
        self.BS1 = ModelB_Bah(dim1,dim2) 
        self.BS2 = ModelB_Bah(dim1,dim2)

        self.conv1 = nn.Conv1d(dim1,dim1,kernel_size=3,padding=1)
        self.fn1 = nn.Linear(dim1*dim2,128)
        self.fn2 = nn.Linear(128,1)
        self.fn3 = nn.Linear(dim1*dim2,128)
        self.fn4 = nn.Linear(128,1)
        self.ac = nn.Sigmoid()
        self.bm1 = nn.BatchNorm1d(dim1)

    def forward(self, x):
        x1,x2 = self.BS1(x)
        x1,x2 = self.BS2(x1)

        x1 = self.bm1(x1)
        x1 = self.conv1(x1)
        x1 = x1.contiguous().view(x1.shape[0],-1)
        x1 = self.fn1(x1)
        x1 = self.fn2(x1)
        
        x2 = x2.contiguous().view(x2.shape[0],-1)
        x2 = self.fn3(x2)
        x2 = self.fn4(x2)
        
        out = x1+x2
        
        out = self.ac(out)
        return out

#BiLSTM+MultiSelf-Attention
class ModelB_MultiSelf(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.1):
        super(ModelB_MultiSelf, self).__init__()
        self.posi = PositionalEncoding(dim1,dim2,dropout)
        self.self_A = MultiSelfAttention(5,dim2)
        self.self_B = MultiSelfAttention(5,dim1)
        self.lstm = nn.LSTM(input_size=dim2,hidden_size=dim2,batch_first=True,bidirectional=False)
        self.conv1 = nn.Conv1d(dim1,dim1,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(dim1,dim1,kernel_size=3,padding=1)
        self.bm1 = nn.BatchNorm1d(dim1)

    def forward(self, x):
        
        x1 = self.conv1(x)
        x1 = self.conv2(x1)  
        x2 = self.posi(x)
        x2,(h_n,c_n) = self.lstm(x2)
        if x2.shape[1] == 1:
            x2 = x2.view(x2.shape[0],x2.shape[2],x2.shape[1])
            x2,attn = self.self_B(x2,x2,x2)
        else:
            x2,attn = self.self_A(x2,x2,x2)
        if x2.shape[2]==1:
            x2 = x2.view(x2.shape[0],x2.shape[2],x2.shape[1])
        out = x1+x2
        #out = x2
        out = self.bm1(out)
        #out = out.view(out.shape[0],-1)
        return out,x2
    
class ModelB_MultiSelf_Pro(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.1):
        super(ModelB_MultiSelf_Pro, self).__init__()
        self.BS1 = ModelB_MultiSelf(dim1,dim2)
        self.BS2 = ModelB_MultiSelf(dim1,dim2)

        self.conv1 = nn.Conv1d(dim1,dim1,kernel_size=3,padding=1)
        self.fn1 = nn.Linear(dim1*dim2,128)
        self.fn2 = nn.Linear(128,1)
        self.fn3 = nn.Linear(dim1*dim2,128)
        self.fn4 = nn.Linear(128,1)
        self.ac = nn.Sigmoid()
        self.bm1 = nn.BatchNorm1d(dim1)

    def forward(self, x):
        x1,x2 = self.BS1(x)
        x1,x2 = self.BS2(x1)

        x1 = self.bm1(x1)
        x1 = self.conv1(x1)
        x1 = x1.contiguous().view(x1.shape[0],-1)
        x1 = self.fn1(x1)
        x1 = self.fn2(x1)
        
        x2 = x2.contiguous().view(x2.shape[0],-1)
        x2 = self.fn3(x2)
        x2 = self.fn4(x2)
        
        #out = x1
        out = x1+x2
        
        out = self.ac(out)
        return out
    
def train(model,data,label,epoch,train_device,model_dir,batch_size):
    if os.path.exists(model_dir+'model.pt'):
        model_train = torch.load(model_dir+'/model.pt')
    else:
        model_train = model
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model_train.parameters(),lr=0.0001)
    #optimizer = torch.optim.Adam(model_train.parameters(),lr=learn_rate)
    #scheduler = StepLR(optimizer,step_size=10,gamma=0.5)
    dataX = torch.Tensor(data).clone().detach()
    label = torch.Tensor(label).clone().detach()
    train_data = TensorDataset(dataX, label)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    running_loss = 0.0
    model_train = model_train.to(train_device)
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target = data
        #inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1])
        inputs = inputs.to(train_device)
        target = target.to(train_device)
        target = target.reshape(target.shape[0],1)
        optimizer.zero_grad()
        outputs = model_train(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx == len(dataX)//batch_size:
            #print('[%d, %5d] epoch loss: %.3f' %(epoch+1,batch_idx+1,running_loss))
            print(running_loss)
    save_model(model_train,model_dir)
    model_train = torch.load(model_dir+'/model.pt')
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    th,_,_,_,_,_,_,_,_,_ = metricsCal.evaluate(model_train,train_loader,train_device)
    return running_loss,th

def test(data,label,best_auc,test_device,model_dir,batch_size,th):
    model_test = load_model(model_dir)
    #model_test.eval()
    #model_test.to(test_device)
    data = torch.Tensor(data).clone().detach()#torch.Tensor(data)
    label = torch.Tensor(label).clone().detach()#.requires_grad_(True)torch.Tensor(label)
    test_data = TensorDataset(data,label)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    _,_,_,_,Sen,Spe, Acc, mcc, AUC = metricsCal.evaluate(model_test,test_loader,test_device,False,th)
    
    print('Accuracy on test set: %d' %Acc)
    print('Sensitivity on test set: %d' %Sen)
    print('Speciality on test set: %d' %Spe)
    print('MCC on test set: %.3f' %mcc)
    print('auc on test set: %.3f' %AUC)
    if(AUC > best_auc):
        torch.save(model_test,model_dir+'model_best.pt')
    return Acc, mcc, AUC

def independTest(data,label,test_device,model_dir,batch_size,th):
    model_test = load_bestModel(model_dir)
    data = torch.Tensor(data).clone().detach()#torch.Tensor(data)
    label = torch.Tensor(label).clone().detach()#.requires_grad_(True)torch.Tensor(label)
    test_data = TensorDataset(data,label)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    _,_,_,_,Sen,Spe, Acc, mcc, AUC = metricsCal.evaluate(model_test,test_loader,test_device,False,th)
    print(Acc,mcc,AUC)
    print('Accuracy on test set: %d %%' %Acc)
    print('Sensitivity on test set: %d %%' %Sen)
    print('Speciality on test set: %d %%' %Spe)
    print('MCC on test set: %.3f' %mcc)
    print('auc on test set: %.3f' %AUC)
    return Acc, mcc, AUC

def independResult(data,label,test_device,model_dir,batch_size,th):
    model_test = load_bestModel(model_dir)
    data = torch.Tensor(data).clone().detach()#torch.Tensor(data)
    label = torch.Tensor(label).clone().detach()#.requires_grad_(True)torch.Tensor(label)
    test_data = TensorDataset(data,label)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    y_score,y_true = metricsCal.evaluate_result(model_test,test_loader,test_device,False,th)
    return y_score,y_true

def analysis_results(pred,label,stack_data,stack_label,th=0.3,strategy="stack"):
    if strategy == "soft":
        clf = SVC(kernel = "rbf",gamma="auto", degree = 1,tol =1e-2, cache_size=7000)
        clf.fit( stack_data, stack_label)
        y_score = clf.predict_proba(pred)
        th,_,_,_,_,_,_,_,_,_ = metricsCal.get_train_metrics( clf.predict_proba(stack_data),stack_label )
        TN, FN, FP, TP, Sen, Spe, Acc, mcc, AUC = metricsCal.get_test_metrics(y_score,label,th)
    elif strategy == "stack":
        y_score = np.mean(pred,axis=1)
        th,_,_,_,_,_,_,_,_,_ = metricsCal.get_train_metrics( y_score,label )
        TN, FN, FP, TP, Sen, Spe, Acc, mcc, AUC = metricsCal.get_test_metrics(y_score,label,th)
    return y_score,TN, FN, FP, TP, Sen, Spe, Acc, mcc, AUC

def load_model(model_dir):
    if os.path.exists(model_dir+'model.pt'):
        model_load = torch.load(model_dir+'model.pt')
    return model_load

def save_model(model_save,model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model_save, model_dir+'model.pt')

def load_bestModel(model_dir):
    if os.path.exists(model_dir+'model_best.pt'):
        model_load = torch.load(model_dir+'model_best.pt',map_location='cuda:0')
    return model_load