import torch
from torch import nn
import pickle
import numpy as np
import math
import copy
import sys
from torch.autograd import Variable
import torch.nn.functional as F
numgpu=2
res=[]
def get_attn():  #获取注意力大小
    return res
class InputEmbeddings(nn.Module):  #嵌入维度，继承nn.module类
    def __init__(self, weight_dict_path):  #初始化
        """
        Inputs:
            weight_dict_path: path of pre-trained embeddings of RNA/dictionary
        """
        super(InputEmbeddings, self).__init__()  #输入维度
        weight_dict = pickle.load(open(weight_dict_path, 'rb'))  #weight_dict,权重字典，读权限

        weights = torch.FloatTensor(np.array(list(weight_dict.values())))  #将读取到的values转化为list的值，变成权重大小
        num_embeddings = len(list(weight_dict.keys()))  #嵌入层数量，通过权重字典的key长度
        embedding_dim = 300  #嵌入维度是300

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)  #定义嵌入层，嵌入层数量和嵌入层维度是原先定义好的,num_embeddings即一个字典里要有多少个词
        self.embedding.weight = nn.Parameter(weights)  #确定好嵌入层的权重值的大小
        self.embedding.weight.requires_grad = False  #嵌入矩阵，这里的grad取False
    def forward(self, x):  #定义计算顺序
        x=x.cuda(numgpu)  #定义GPU块
        out = self.embedding(x.type(torch.cuda.LongTensor))  #将x转化为LongTensor类型，然后放入嵌入层
        return out  #返回输出值


class PositionalEncoding(nn.Module):  #定义位置编码类
    def __init__(self, embedding_dim, dropout, max_len=5000):  #嵌入维度，dropout率，最大长度
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  #定义Dropout层

        pe = torch.zeros(max_len, embedding_dim)  #定义最大长度*嵌入维度的0矩阵，即5000*300

        position = torch.arange(0., max_len).unsqueeze(1)  # [max_len, 1], 位置编码，0-4999
        div_term = torch.exp(torch.arange(0., embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))  #步长为2，总共150个数

        pe[:, 0::2] = torch.sin(position * div_term)  #奇数取sin，偶数取cos
        pe[:, 1::2] = torch.cos(position * div_term)  #奇数、偶数分开处理，位置矩阵
        pe = pe.unsqueeze(0)  # 增加维度

        self.register_buffer('pe', pe)  # 内存中定一个常量，模型保存和加载的时候，可以写入和读出

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)  # Embedding + PositionalEncoding,x的值加上了位置转移矩阵
        return self.dropout(x)  #添加Dropout层后的结果


def clones(module, N):  #定义clones方法
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])  #让原来变量不影响,且克隆module N次

def attention(query, key, value, mask=None, dropout=None):  # q,k,v: [batch, h, seq_len, d_k]
    d_k = query.size(-1)  # query的维度
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 打分机制 [batch, h, seq_len, seq_len]

    p_atten = F.softmax(scores, dim=-1)  # 对最后一个维度归一化得分, [batch, h, seq_len, seq_len]

    if dropout is not None:
        p_atten = dropout(p_atten)
    return torch.matmul(p_atten, value), p_atten  # [batch, h, seq_len, d_k] 作矩阵的乘法
#输入三个矩阵向量，获取atten权值

# 建立一个全连接的网络结构
class MultiHeadedAttention(nn.Module):  #多头注意力机制
    def __init__(self, h, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim % h == 0  #断言，可以整除

        self.d_k = embedding_dim // h  # 将 embedding_dim 分割成 h份 后的维度
        self.h = h  # h 指的是 head数量
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)  #克隆四份Linear网络层

        self.dropout = nn.Dropout(p=dropout)  #定义Dropout层

    def forward(self, query, key, value, mask=None):  # q,k,v: [batch, seq_len, embedding_dim]

        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch, seq_len, 1]
        nbatches = query.size(0)  #批数量

        # 1. Do all the linear projections(线性预测) in batch from embeddding_dim => h x d_k
        # [batch, seq_len, h, d_k] -> [batch, h, seq_len, d_k]
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  #转换形式大小等
                             for l, x in zip(self.linears, (query, key, value))]  #获取zip的query,key,value权重矩阵

        # 2. Apply attention on all the projected vectors in batch.
        # atten:[batch, h, seq_len, d_k], p_atten: [batch, h, seq_len, seq_len]
        attn, p_atten = attention(query, key, value, mask=mask, dropout=self.dropout)  #得到了三个矩阵之后，返回权值
        # get p_atten
        # res.append(p_atten.cpu().detach().numpy())

        # 3. "Concat" using a view and apply a final linear.
        # [batch, h, seq_len, d_k]->[batch, seq_len, embedding_dim]
        attn = attn.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)  #定义attn值
        out=self.linears[-1](attn)  #得到最后一层线性层的输出
        return out  #返回out结果


class MyTransformerModel(nn.Module):  #定义Transformer类
    def __init__(self, embedding_dim, p_drop, h, output_size):
        super(MyTransformerModel, self).__init__()
        self.drop = nn.Dropout(p_drop)

        # Embeddings,
        self.embeddings = InputEmbeddings('../Embeddings/embeddings_12RM.pkl')
        # H: [e_x1 + p_1, e_x2 + p_2, ....]
        self.position = PositionalEncoding(embedding_dim, p_drop)
        # Multi-Head Attention
        self.atten = MultiHeadedAttention(h, embedding_dim)  # self-attention-->建立一个全连接的网络结构
        # 层归一化(LayerNorm)
        self.norm = nn.LayerNorm(embedding_dim)
        # self.linear1=nn.Linear(embedding_dim,512)
        # self.linear2=nn.Linear(512,512)
        # self.linear3=nn.Linear(512,embedding_dim)
        # Feed Forward
        self.linear = nn.Linear(embedding_dim, output_size)
        # 初始化参数
        self.init_weights()
    def init_weights(self):
        init_range = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)
    def forward(self, inputs):  # 维度均为: [batch, seq_len]

        embeded = self.embeddings(inputs)  # 1. InputEmbedding [batch, seq_len, embedding_dim]
        #         print(embeded.shape)              # torch.Size([36, 104, 100])

        embeded = self.position(embeded)  # 2. PosionalEncoding [batch, seq_len, embedding_dim]
        #         print(embeded.shape)              # torch.Size([36, 104, 100])

        # mask = mask.unsqueeze(2)  # [batch, seq_len, 1]

        # 3.1 MultiHeadedAttention [batch, seq_len. embedding_dim]
        inp_atten = self.atten(embeded, embeded, embeded)
        # 3.2 LayerNorm [batch, seq_len, embedding_dim]
        inp_atten = self.norm(inp_atten + embeded)
        #         print(inp_atten.shape)           
        # opm=self.linear1(inp_atten)
        # opm=self.linear2(opm)
        # opm=self.linear3(opm)
        inp_atten=self.norm(inp_atten)

        #         print(inp_atten.sum(1).shape, mask.sum(1).shape)  # [batch, emb_dim], [batch, 1]
        b_avg = inp_atten.sum(1) / (embeded.shape[1] + 1e-5)  # [batch, embedding_dim]

        return self.linear(b_avg).squeeze()  # [batch, 1] -> [batch]
# *******************************************************************************************************************************************************
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
        hidden_with_time_axis = torch.unsqueeze(hidden_states,dim=1)
        score  = self.V(nn.Tanh()(self.W1(values)+self.W2(hidden_with_time_axis)))
        attention_weights = nn.Softmax(dim=1)(score)
        values = torch.transpose(values,1,2)   # transpose to make it suitable for matrix multiplication
        #print(attention_weights.shape,values.shape)
        context_vector = torch.matmul(values,attention_weights)
        context_vector = torch.transpose(context_vector,1,2)
        return context_vector, attention_weights

class EmbeddingSeq(nn.Module):
    def __init__(self,weight_dict_path):
        """
        Inputs:
            weight_dict_path: path of pre-trained embeddings of RNA/dictionary
        """
        super(EmbeddingSeq,self).__init__()
        weight_dict = pickle.load(open(weight_dict_path,'rb'))

        weights = torch.FloatTensor(list(weight_dict.values())).cuda()
        num_embeddings = len(list(weight_dict.keys()))
        embedding_dim = 300

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,embedding_dim=embedding_dim)
        self.embedding.weight = nn.Parameter(weights)
        self.embedding.weight.requires_grad = False

    def forward(self,x):

        out = self.embedding(x.type(torch.cuda.LongTensor))

        return out

class EmbeddingHmm(nn.Module):
    def __init__(self,t,out_dims):
        """
        Inputs:
            length: the length of input sequence
            t: the hyperparameters used for parallel message update iterations
            out_dims: dimension of new embedding
        """
        super(EmbeddingHmm,self).__init__()

        self.T = t
        self.out_dims = out_dims
        self.W1 = nn.Linear(4,out_dims)
        self.W2 = nn.Linear(out_dims,out_dims)
        self.W3 = nn.Linear(4,out_dims)
        self.W4 = nn.Linear(out_dims,out_dims)
        self.relu = nn.ReLU()

    def forward(self,x):
        """
        Inputs:
            x: RNA/DNA sequences using one-hot encoding, channel first: (bs,dims,seq_len)
        Outputs:
            miu: hmm encoding of RNA/DNA, channel last: (bs,seq_len,dims)
        """
        batch_size,length = x.shape[0], x.shape[-1]
        V = torch.zeros((batch_size,self.T+1,length+2,length+2,self.out_dims)).cuda()
        for i in range(1,self.T+1):
            for j in range(1,length+1):
                V[:,i,j,j+1,:] = self.relu(self.W1(x[:,:,j-1].clone())+self.W2(V[:,i-1,j-1,j,:].clone()))
                V[:,i,j,j-1,:] = self.relu(self.W1(x[:,:,j-1].clone())+self.W2(V[:,i-1,j+1,j,:].clone()))
        miu = torch.zeros((batch_size,length,self.out_dims)).cuda()

        for i in range(1,length+1):
            miu[:,i-1,:]= self.relu(self.W3(x[:,:,i-1].clone())+self.W4(V[:,self.T,i-1,i].clone())+self.W4(V[:,self.T,i+1,i].clone()))
        return miu

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, num_task):
        super(MultiTaskLossWrapper, self).__init__()
        self.num_task = num_task
        self.log_vars = nn.Parameter(torch.zeros((num_task)))

    def forward(self,  y_pred,targets):
        num_examples=targets.shape[0]
        k=0.7
        def binary_cross_entropy(x, y):
            loss = -(torch.log(x+10e-9) * y + torch.log(1 - x+10e-9) * (1 - y))
            return torch.sum(loss)
        # loss = nn.BCELoss(reduction='sum') fail to double backwards
        loss_output = 0
        #if ohem
        # loss_output = torch.zeros(num_examples).cuda(numgpu)
        for i in range(self.num_task):
            out = torch.exp(-self.log_vars[i])*binary_cross_entropy(y_pred[i],targets[:,i]) + self.log_vars[i]
            if math.isnan(out):
                print(y_pred[i],targets[:,i])
                sys.exit()
            loss_output += out
        # if ohem
        # val, idx = torch.topk(loss_output,int(k*num_examples))
        # loss_output[loss_output<val[-1]] = 0
        # loss = torch.sum(loss_output)

        return loss_output
