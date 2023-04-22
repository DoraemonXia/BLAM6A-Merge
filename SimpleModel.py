import numpy as np
import pandas as pd
import torch
from torch import nn
import re
from sklearn.utils import shuffle
from utils import metricsCal_MidFusion as metricsCal
from torch.utils.data import DataLoader,TensorDataset
import math
import sys
import copy
import pickle
from torch.autograd import Variable
from sklearn.model_selection import KFold
import torch.nn.functional as F
import os

def attention(query, key, value, mask=None, dropout=None):  # q,k,v: [batch, h, seq_len, d_k]
    d_k = query.size(-1)  # query的维度
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 打分机制 [batch, h, seq_len, seq_len]
    p_atten = F.softmax(scores, dim=-1)  # 对最后一个维度归一化得分, [batch, h, seq_len, seq_len]
    if dropout is not None:
        p_atten = dropout(p_atten)
    return torch.matmul(p_atten, value), p_atten  # [batch, h, seq_len, d_k] 作矩阵的乘法

class PositionalEncoding(nn.Module):

    def __init__(self, dim1, dim2, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        #if dim % 2 != 0:
        #    raise ValueError("Cannot use sin/cos positional encoding with "
        #                     "odd dim (got dim={:d})".format(dim))

        """
        构建位置编码pe
        pe公式为：
        PE(pos,2i/2i+1) = sin/cos(pos/10000^{2i/d_{model}})
        """
        pe = torch.zeros(max_len, dim2)  # max_len 是解码器生成句子的最长的长度，假设是 10
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
            emb = torch.tensor(emb) + self.pe[:,:emb.shape[1]]
        else:
            emb = emb + self.pe[step]
        #emb = self.drop_out(emb)
        emb = self.bm1(emb.to(torch.float32))
        return emb
    
def clones(module, N):  #定义clones方法
    return nn.ModuleList([copy.deepcopy(module)
                          for _ in range(N)])  #让原来变量不影响,且克隆module N次

class SelfAttention(nn.Module):  #多头注意力机制

    def __init__(self,embedding_dim, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)  #克隆四份Linear网络层
        self.dropout = nn.Dropout(p=dropout)  #定义Dropout层

    def forward(self,query,key,value,mask=None):  # q,k,v: [batch, seq_len, embedding_dim]
        nbatches = query.shape[0]  #批数量
        query, key, value = [
            l(x) for l, x in zip(self.linears,
                            (query.to(torch.float32),
                             key.to(torch.float32),
                             value.to(torch.float32)))
        ]  #获取zip的query,key,value权重矩阵
        attn, p_atten = attention(query,key,value,mask=mask,dropout=self.dropout)
        out = self.linears[-1](attn)  #得到最后一层线性层的输出
        return out,p_atten  #返回out结果
    
class MultiSelfAttention(nn.Module):  #多头自注意力机制
    def __init__(self, h,embedding_dim ,dropout=0.1):
        super(MultiSelfAttention, self).__init__()
        self.h = h
        self.attn_modules = clones(SelfAttention(embedding_dim), h)  #克隆h份Linear网络层 

    def forward(self,query,key,value,mask=None):  # q,k,v: [batch, seq_len, embedding_dim]             
        for i in range(self.h):   
            if i != 0:
                out_one,attn_one= self.attn_modules[i](query,key,value)
                out = torch.add(out,torch.tensor(out_one))
                attn = torch.add(attn,torch.tensor(attn_one))
            else:
                out,attn = self.attn_modules[i](query,key,value)
        return out,attn  #返回out结果 

class MultiHeadAttention(nn.Module):  #多头注意力机制
    def __init__(self, h, embedding_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_k = embedding_dim // h  # 将 embedding_dim 分割成 h份 后的维度
        self.h = h  # h 指的是 head数量
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)  #克隆四份Linear网络层      
        self.dropout = nn.Dropout(p=dropout)  #定义Dropout层

    def forward(self,query,key,value,mask=None):  # q,k,v: [batch, seq_len, embedding_dim]
        nbatches = query.shape[0]  #批数量
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  #转换形式大小等
            for l, x in zip(self.linears,
                            (query.to(torch.float32),
                             key.to(torch.float32), 
                             value.to(torch.float32) ) )
        ]  #获取zip的query,key,value权重矩阵
        attn, p_atten = attention(query,key,value,mask=mask,dropout=self.dropout)
        attn = attn.transpose(1,2).contiguous().view(nbatches, -1,self.h * self.d_k)  #定义attn值
        out = self.linears[-1](attn)  #得到最后一层线性层的输出
        return out,attn  #返回out结果
    
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
        self.bm1 = nn.BatchNorm1d(dim1)

    def forward(self, x):
        x2 = self.posi(x)
        x2,(h_n,c_n) = self.lstm(x2)
        if x2.shape[1] == 1:
            x2 = x2.view(x2.shape[0],x2.shape[2],x2.shape[1])
            x2,attn = self.self_B(x2,x2,x2)
        else:
            x2,attn = self.self_A(x2,x2,x2)
        if x2.shape[2]==1:
            x2 = x2.view(x2.shape[0],x2.shape[2],x2.shape[1])
        out = self.bm1(x2)
        return out

class ModelB_MultiSelf(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.1):
        super(ModelB_MultiSelf, self).__init__()
        self.posi = PositionalEncoding(dim1,dim2,dropout)
        self.self_A = MultiSelfAttention(5,dim2)
        self.self_B = MultiSelfAttention(5,dim1)
        self.lstm = nn.LSTM(input_size=dim2,hidden_size=dim2,batch_first=True,bidirectional=False)
        self.bm1 = nn.BatchNorm1d(dim1)

    def forward(self, x):
        x2 = self.posi(x)
        x2,(h_n,c_n) = self.lstm(x2)
        if x2.shape[1] == 1:
            x2 = x2.view(x2.shape[0],x2.shape[2],x2.shape[1])
            x2,attn = self.self_B(x2,x2,x2)
        else:
            x2,attn = self.self_A(x2,x2,x2)
        if x2.shape[2]==1:
            x2 = x2.view(x2.shape[0],x2.shape[2],x2.shape[1])
        out = self.bm1(x2)
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
        out = self.bm1(x2)
        return out

#dim1 dim2 emb
#dim3 dim4 PSNP
#dim5 dim6 PCP
#dim7 dim8 DBPF
class Model_Mid_Fusion_Pro(nn.Module):
    def __init__(self, dim1,dim2,dim3,dim4,dim5,dim6,dim7,dim8, dropout=0.1):
        super(Model_Mid_Fusion_Pro, self).__init__()
        
        self.BBah = ModelB_Bah(dim7,dim8) 
        self.BMS_emb = ModelB_MultiSelf(dim1,dim2)
        self.BMS_PSNP = ModelB_MultiSelf(dim3,dim4)
        self.BS = ModelBS(dim5,dim6)
        
        self.conv1 = nn.Conv1d(1,1,kernel_size=3,padding=1)
        self.fn1 = nn.Linear(dim1*dim2+dim3*dim4+dim5*dim6+dim7*dim8,128)
        self.fn2 = nn.Linear(128,1)
        self.ac = nn.Sigmoid()
        self.bm1 = nn.BatchNorm1d(dim1)

    def forward(self, emb,PSNP,PCP,DBPF):
        x_emb = self.BMS_emb(emb)
        x_PSNP = self.BMS_PSNP(PSNP)
        x_PCP = self.BS(PCP)
        x_DBPF = self.BBah(DBPF)
        x1 = torch.concat([x_emb.contiguous().view(x_emb.shape[0],-1),x_PSNP.contiguous().view(x_PSNP.shape[0],-1),
                     x_PCP.contiguous().view(x_PCP.shape[0],-1),x_DBPF.contiguous().view(x_DBPF.shape[0],-1)],1)
        x1 = x1.contiguous().view(x1.shape[0],1,x1.shape[1])
        x1 = self.conv1(x1)
        x1 = x1.contiguous().view(x1.shape[0],-1)
        x1 = self.fn1(x1)
        x1 = self.fn2(x1)
        out = self.ac(x1)
        return out
    
class Model_Mid_Fn_Fusion_Pro(nn.Module):
    def __init__(self, dim1,dim2,dim3,dim4,dim5,dim6,dim7,dim8, dropout=0.1):
        super(Model_Mid_Fn_Fusion_Pro, self).__init__()
        
        self.BBah = ModelB_Bah(dim7,dim8) 
        self.BMS_emb = ModelB_MultiSelf(dim1,dim2)
        self.BMS_PSNP = ModelB_MultiSelf(dim3,dim4)
        self.BS = ModelBS(dim5,dim6)
        
        self.conv1 = nn.Conv1d(dim1,dim1,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(dim3,dim3,kernel_size=3,padding=1)
        self.conv3 = nn.Conv1d(dim5,dim5,kernel_size=3,padding=1)
        self.conv4 = nn.Conv1d(dim7,dim7,kernel_size=3,padding=1)
        self.fn1 = nn.Linear(dim1*dim2,128)
        self.fn2 = nn.Linear(dim3*dim4,128)
        self.fn3 = nn.Linear(dim5*dim6,128)
        self.fn4 = nn.Linear(dim7*dim8,128)
        
        self.fn5 = nn.Linear(512,32)
        self.fn6 = nn.Linear(32,1)
        self.ac = nn.Sigmoid()
        self.bm1 = nn.BatchNorm1d(dim1)
        self.bm2 = nn.BatchNorm1d(dim3)
        self.bm3 = nn.BatchNorm1d(dim5)
        self.bm4 = nn.BatchNorm1d(dim7)

    def forward(self, emb,PSNP,PCP,DBPF):
        x_emb = self.BMS_emb(emb)
        x_PSNP = self.BMS_PSNP(PSNP)
        x_PCP = self.BS(PCP)
        x_DBPF = self.BBah(DBPF)
        
        x_emb = self.bm1(x_emb)
        x_PSNP = self.bm2(x_PSNP)
        x_PCP = self.bm3(x_PCP)
        x_DBPF = self.bm4(x_DBPF)
        
        x_emb = self.conv1(x_emb)
        x_emb = x_emb.contiguous().view(x_emb.shape[0],-1)
        x_PSNP = self.conv2(x_PSNP)
        x_PSNP = x_PSNP.contiguous().view(x_PSNP.shape[0],-1)
        x_PCP = self.conv3(x_PCP)
        x_PCP = x_PCP.contiguous().view(x_PCP.shape[0],-1)
        x_DBPF = self.conv4(x_DBPF)
        x_DBPF = x_DBPF.contiguous().view(x_DBPF.shape[0],-1)
        
        x_emb = self.fn1(x_emb)
        x_PSNP = self.fn2(x_PSNP)
        x_PCP = self.fn3(x_PCP)
        x_DBPF = self.fn4(x_DBPF)
        
        x1 = torch.concat([x_emb,x_PSNP,x_PCP,x_DBPF],1)
        x1 = self.fn5(x1)
        x1 = self.fn6(x1)
        out = self.ac(x1)
        return out
    
class ModelBS_Pro(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.1):
        super(ModelBS_Pro, self).__init__()
        self.BS1 = ModelBS(dim1,dim2) 

        self.conv1 = nn.Conv1d(dim1,dim1,kernel_size=3,padding=1)
        self.fn1 = nn.Linear(dim1*dim2,128)
        self.fn2 = nn.Linear(128,1)
        self.ac = nn.Sigmoid()
        self.bm1 = nn.BatchNorm1d(dim1)

    def forward(self, x):
        x1 = self.BS1(x)
        x1 = self.bm1(x1)
        x1 = self.conv1(x1)
        x1 = x1.contiguous().view(x1.shape[0],-1)
        x1 = self.fn1(x1)
        x1 = self.fn2(x1)
        out = self.ac(x1)
        return out
    
#BiLSTM+Self-Attention
class ModelBS_noPos(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.1):
        super(ModelBS_noPos, self).__init__()
        self.posi = PositionalEncoding(dim1,dim2,dropout)
        self.self_A = SelfAttention(dim2)
        self.self_B = SelfAttention(dim1)
        self.lstm = nn.LSTM(input_size=dim2,hidden_size=dim2,batch_first=True,bidirectional=False)
        self.bm1 = nn.BatchNorm1d(dim1)

    def forward(self, x):
        x2,(h_n,c_n) = self.lstm(x)
        if x2.shape[1] == 1:
            x2 = x2.view(x2.shape[0],x2.shape[2],x2.shape[1])
            x2,attn = self.self_B(x2,x2,x2)
        else:
            x2,attn = self.self_A(x2,x2,x2)
        if x2.shape[2]==1:
            x2 = x2.view(x2.shape[0],x2.shape[2],x2.shape[1])
        out = self.bm1(x2)
        return out
    
class ModelBS_Pro_noPos(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.1):
        super(ModelBS_Pro_noPos, self).__init__()
        self.BS1 = ModelBS_noPos(dim1,dim2) 
        self.conv1 = nn.Conv1d(dim1,dim1,kernel_size=3,padding=1)
        self.fn1 = nn.Linear(dim1*dim2,128)
        self.fn2 = nn.Linear(128,1)
        self.ac = nn.Sigmoid()
        self.bm1 = nn.BatchNorm1d(dim1)

    def forward(self, x):
        x1 = self.BS1(x)
        x1 = self.bm1(x1)
        x1 = self.conv1(x1)
        x1 = x1.contiguous().view(x1.shape[0],-1)
        x1 = self.fn1(x1)
        x1 = self.fn2(x1)
        out = self.ac(x1)
        return out 
    
def train(model,data,label,epoch,train_device,model_dir,batch_size):
    
    if os.path.exists(model_dir+'model.pt'):
        model_train = torch.load(model_dir+'/model.pt')
    else:
        model_train = model
        
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model_train.parameters(),lr=0.001)  #改变学习率
    
    dataX = torch.Tensor(data).clone().detach()
    label = torch.Tensor(label).clone().detach()
    train_data = TensorDataset(dataX, label)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    
    
    running_loss = 0.0
    model_train = model_train.to(train_device)
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target = data
        #inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1])
        #inputs = inputs.to(train_device)
        target = target.to(train_device)
        target = target.reshape(target.shape[0],1)
        optimizer.zero_grad()
        inputs_emb = inputs[:,:,:3800].reshape(inputs.shape[0],38,100)
        inputs_PSNP = inputs[:,:,3800:3841].reshape(inputs.shape[0],1,41)
        inputs_PCP = inputs[:,:,3841:4241].reshape(inputs.shape[0],1,400)
        inputs_DBPF = inputs[:,:,4241:4441].reshape(inputs.shape[0],40,5)
        inputs_emb = inputs_emb.to(train_device)
        inputs_PSNP = inputs_PSNP.to(train_device)
        inputs_PCP = inputs_PCP.to(train_device)
        inputs_DBPF = inputs_DBPF.to(train_device)
        
        outputs = model_train(inputs_emb,inputs_PSNP,inputs_PCP,inputs_DBPF)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx == len(dataX)//batch_size:
            #print('[%d, %5d] epoch loss: %.3f' %(epoch+1,batch_idx+1,running_loss))
            print(running_loss)
    save_model(model_train,model_dir)
    model_train = torch.load(model_dir+'/model.pt')
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 选择设备
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
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 选择设备
    _,_,_,_,Sen,Spe, Acc, mcc, AUC = metricsCal.evaluate(model_test,test_loader,test_device,False,th)
    
    print('Accuracy on test set: %d %%' %Acc)
    print('Sensitivity on test set: %d %%' %Sen)
    print('Speciality on test set: %d %%' %Spe)
    print('MCC on test set: %.3f' %mcc)
    print('auc on test set: %.3f' %AUC)
    if(AUC >= best_auc):
        torch.save(model_test,model_dir+'model_best.pt')
    return Acc, mcc, AUC

def independTest(data,label,test_device,model_dir,batch_size,th):
    model_test = load_bestModel(model_dir)
    data = torch.Tensor(data).clone().detach()#torch.Tensor(data)
    label = torch.Tensor(label).clone().detach()#.requires_grad_(True)torch.Tensor(label)
    test_data = TensorDataset(data,label)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 选择设备
    _,_,_,_,Sen,Spe, Acc, mcc, AUC = metricsCal.evaluate(model_test,test_loader,test_device,False,th)
    print(Acc,mcc,AUC)
    print('Accuracy on test set: %d %%' %Acc)
    print('Sensitivity on test set: %d %%' %Sen)
    print('Speciality on test set: %d %%' %Spe)
    print('MCC on test set: %.3f' %mcc)
    print('auc on test set: %.3f' %AUC)
    return Acc, mcc, AUC

def independResult(data,label,test_device,model_dir,batch_size,th=0.5):
    model_test = load_bestModel(model_dir)
    data = torch.Tensor(data).clone().detach()#torch.Tensor(data)
    label = torch.Tensor(label).clone().detach()#.requires_grad_(True)torch.Tensor(label)
    test_data = TensorDataset(data,label)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 选择设备
    y_score,y_true = metricsCal.evaluate_result(model_test,test_loader,test_device,False,th)
    return y_score,y_true

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