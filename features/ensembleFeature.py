import pandas as pd
import numpy as np
#import torch
import os
import math
from collections import Counter
from itertools import product
import collections
from sklearn.utils import shuffle
import itertools
from gensim.models import word2vec

##############################
#No:1--NCP+ANF     length*4
#No:2--PSNP       (length-window+1)*1
#No:3--Binary      length*4
#No:4--emb_seqs    38*100
#No:5--ENAC       length*(input_length-window_size+1)*4   
#No:6--PseDNC     length*(input_length-window_size+1)*4    
#No:7--DNC       16
#No:8--EIIP      length*1
#No:9--PCP      length*(input_length-window_size+1)*4   CP
#No:10--DBPF      length*(input_length-2+1)*5

#************—————————How to use—————————*************
#--1--*************NCPA(sequences)************************************
#--2--*************PSNP(trainPos,trainNeg,testPos,testNeg,k=1)********
#--3--*************Binary(sequences)**********************************
#--4--*************emb_seqs(sequences)********************************
######################################################################

#NCPA
#shape: batch_Size*input_length*4
def calculate(sequence):
    X = []
    dictNum = {'A' : 0, 'T' : 0, 'C' : 0, 'G' : 0, 'N': 0};
    for i in range(len(sequence)):
        if sequence[i] in dictNum.keys():
            dictNum[sequence[i]] += 1;
            X.append(dictNum[sequence[i]] / float(i + 1));
    return np.array(X)

def NCPA(sequences):
    chemical_property = {
        'A': [1, 1, 1],
        'T': [0, 1, 0],
        'G': [1, 0, 0],
        'C': [0, 0, 1]
    }
    ncp_feature = []
    for seq in sequences:
        ncp = []
        for aaindex, aa in enumerate(seq):
            ncp.append(chemical_property.get(aa, [0, 0, 0]))
        ncpa=np.append(ncp,calculate(seq).reshape(-1,1),axis=1)
        ncp_feature.append(ncpa)
    return np.array(ncp_feature)

#PSNP
def CalculateMatrix(data, order, k):
    if k == 1:        
        matrix = np.zeros((len(data[0]), 4))
        for i in range(len(data[0])): 
            for j in range(len(data)):  
                matrix[i][order[data[j][i:i+1]]] += 1     
    return matrix

def PSNP(trainPos,trainNeg,testPos,testNeg,k=1):
    #String
    train_positive = []
    for pos in trainPos:
        train_positive.append(str(pos))
    train_negative = []
    for neg in trainNeg:
        train_negative.append(str(neg))
    train_p_num = len(train_positive)
    train_n_num = len(train_negative)
    
    test_positive = []
    for pos in testPos:
        test_positive.append(str(pos))
    test_negative = []
    for neg in testNeg:
        test_negative.append(str(neg))
   
    test_p_num = len(test_positive)
    test_n_num = len(test_negative)
    
    test_lp = len(test_positive[0])
    test_ln = len(test_negative[0])
    
    nucleotides = ['A', 'C', 'G', 'T'] 
    
    if k == 1 :
        nuc = [n1 for n1 in nucleotides]
        order = {}
        for i in range(len(nuc)):
            order[nuc[i]] = i 
        matrix_po = CalculateMatrix(train_positive, order, 1) 
        matrix_ne = CalculateMatrix(train_negative, order, 1)

        F1 = matrix_po/train_p_num 
        F2 = matrix_ne/train_n_num       
        F = F1 - F2  

        testPosCode = []
        for sequence in test_positive:  
            for j in range(len(sequence)):                
                po_number = F[j][order[sequence[j:j+1]]]
                testPosCode.append(po_number)  
        testPosCode = np.array(testPosCode)
        testPosCode = testPosCode.reshape((test_p_num,test_lp)) 
        
        testNegCode = []    
        for sequence in test_negative:    
            for i in range(len(sequence)):
                ne_number = F[i][order[sequence[i:i+1]]]
                testNegCode.append(ne_number)  
        testNegCode = np.array(testNegCode)
        testNegCode = testNegCode.reshape((test_n_num,test_ln))  
        
        trainPosCode = []    
        for sequence in train_positive:    
            for i in range(len(sequence)):
                po_number = F[i][order[sequence[i:i+1]]]
                trainPosCode.append(po_number)  
        trainPosCode = np.array(trainPosCode)
        trainPosCode = trainPosCode.reshape((train_p_num,test_ln)) 
        
        trainNegCode = []    
        for sequence in train_negative:    
            for i in range(len(sequence)):
                ne_number = F[i][order[sequence[i:i+1]]]
                trainNegCode.append(ne_number)  
        trainNegCode = np.array(trainNegCode)
        trainNegCode = trainNegCode.reshape((train_n_num,test_ln))
        
    return trainPosCode,trainNegCode,testPosCode,testNegCode

#batch_Size*input_length*4
def Binary(sequences):
    AA = 'ACGT'
    binary_feature = []
    for seq in sequences:
        # seq=str(seq)[2:23]
        binary = []
        for aa in seq:
            binary_one = []
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                binary_one.append(tag)
            #binary.append(binary_one)
            binary.append(binary_one)
        binary_feature.append(binary)
    return np.array(binary_feature)

#word2vec
def emb_seqs(sequences, features=100, num = 4):
    w2v_model = word2vec.Word2Vec.load("features/dna_w2v_100.pt")
    seqs_emb = []
    for seq in sequences:
        seq_emb = []
        for i in range(len(seq) - num + 1):
            try:
                seq_emb.append(np.array(w2v_model.wv[seq[i:i+num]]))
            except:
                seq_emb.append(np.array(np.zeros([features])))
        seqs_emb.append(seq_emb)
    seqs_emb = np.array(seqs_emb).reshape(len(seqs_emb),-1,features)
    return seqs_emb #

#batch_size*(input_length-window+1)*4
def ENAC(sequences):
    AA = 'ACGT'
    enac_feature = []
    window = 5
    for seq in sequences:
        #seq=str(seq)[2:23]
        l = len(seq)
        enac = []
        for i in range(0, l):
            if i < l and i + window <= l:
                enac_one = []
                count = Counter(seq[i:i + window])
                for key in count:
                    count[key] = count[key] / len(seq[i:i + window])
                for aa in AA:
                    enac_one.append(count[aa])
                    #enac+=count[aa]
                enac.append(enac_one)  #
                #enac += enac_one  #
        enac_feature.append(enac)
    return np.array(enac_feature)

def PseDNC(sequences):
    gene_type='DNA'
    fill_NA='0'
    propertyname=r"physical_chemical_properties_DNA.txt"
    
    phisical_chemical_proporties=pd.read_csv(propertyname,header=None,index_col=None)
    
    DNC_key=phisical_chemical_proporties.values[:,0]
    if fill_NA=="1":
        DNC_key[21]='NA'
    
    DNC_value=phisical_chemical_proporties.values[:,1:]
    DNC_value=np.array(DNC_value).T  #
    DNC_value_scale=[[]]*len(DNC_value)  
    for i in range(len(DNC_value)):
        average_=sum(DNC_value[i]*1.0/len(DNC_value[i]))  #
        std_=np.std(DNC_value[i],ddof=1)  #
        DNC_value_scale[i]=[round((e-average_)/std_,2) for e in DNC_value[i]]  #
    DNC_value_scale=list(zip(*DNC_value_scale))  #

    DNC_len=len(DNC_value_scale)  #
    
    w=0.9
    Lamda=6  #
    result_value=[]
    m6a_len=len(sequences[0])  #
    
    m6a_num=len(sequences)  #
    for m6a_line_index in range(m6a_num):  #
        frequency=[0]*len(DNC_key)  #
        #print len(frequency)
        m6a_DNC_value=[[]]*(m6a_len-1)  #
        #print m6a_DNC_value
        for m6a_line_doublechar_index in range(m6a_len):
            for DNC_index in range(len(DNC_key)):
                if sequences[m6a_line_index][m6a_line_doublechar_index:m6a_line_doublechar_index+2]==DNC_key[DNC_index]:
                    #print m6aseq[2][0:2]
                    m6a_DNC_value[m6a_line_doublechar_index]=DNC_value_scale[DNC_index]  #
                    frequency[DNC_index]+=1  #
        #print m6a_DNC_value

        frequency=[e/float(sum(frequency)) for e in frequency]  #
        p=sum((frequency))  #
        
        one_line_value_with = 0.0
        sita = [0] * Lamda  #
        for lambda_index in range(1, Lamda + 1):
            one_line_value_without_ = 0.0
            for m6a_sequence_value_index in range(1, m6a_len - lambda_index):
                temp = list(map(lambda x,y : round((x - y) ** 2,8), list(np.array(m6a_DNC_value[m6a_sequence_value_index - 1])),list(np.array(m6a_DNC_value[m6a_sequence_value_index - 1 + lambda_index]))))

                temp_value = round(sum(temp) * 1.0 / DNC_len,8)
                one_line_value_without_ += temp_value
            one_line_value_without_ = round(one_line_value_without_ / (m6a_len - lambda_index-1),8)
            sita[lambda_index - 1] = one_line_value_without_
            one_line_value_with += one_line_value_without_
        dim = [0] * (len(DNC_key) + Lamda)
        for index in range(1, len(DNC_key) + Lamda+1):
            if index <= len(DNC_key):
                dim[index - 1] = frequency[index - 1] / (1.0 + w * one_line_value_with)
            else:
                dim[index - 1] = w * sita[index - len(DNC_key)-1] / (1.0 + w * one_line_value_with)
            dim[index-1]=round(dim[index-1],8)
        result_value.append(dim)
    return np.array(result_value)

def query(short,sequence):
    count = 0
    for i in range(len(sequence)-len(short)+1):
        if sequence[i:i+len(short)]==short:
            count+=1
    return count
def DNC(sequences):
    final = []
    for seq in sequences:
        seq_length = len(seq)-1
        RNA = ['A', 'T', 'C', 'G']
        di_nucleotide_values = []
        di_nucleotide_dict = {"".join(i): 0 for i in product(RNA, repeat=2)}
        for di_nucleotide in di_nucleotide_dict.keys():
            di_nucleotide_dict[di_nucleotide] = round(query(di_nucleotide,seq) / seq_length, 3)
        for dict_value in di_nucleotide_dict.values():
            di_nucleotide_values.append(dict_value)
        final.append(di_nucleotide_values)
    return np.array(final)

def EIIP(sequences):
    dic = {"A": 0.1260,"C": 0.1340,"G": 0.0806,"T": 0.1335}
    result = []
    for seq in sequences:
        result_one = []
        for k in seq:
            result_one.append(dic[k])
        result.append(result_one)
    return np.array(result)

def PCP(sequences):
    path=""
    gene_type="DNA"
    fill_NA='0'
    propertyname="features/physical_chemical_properties_DNA.txt"
    physical_chemical_properties_path=propertyname

    data=pd.read_csv(physical_chemical_properties_path,header=None,index_col=None)#read the phisical chemichy proporties
    prop_key=data.values[:,0]

    if fill_NA=="1":
        prop_key[21]='NA'
    prop_data=data.values[:,1:]
    prop_data=np.matrix(prop_data)
    DNC_value=np.array(prop_data).T
    DNC_value_scale=[[]]*len(DNC_value)
    for i in list(range(len(DNC_value))):
        average_=sum(DNC_value[i]*1.0/len(DNC_value[i]))
        std_=np.std(DNC_value[i],ddof=1)
        DNC_value_scale[i]=[round((e-average_)/std_,2) for e in DNC_value[i]]
    prop_data_transformed=list(zip(*DNC_value_scale))
    prop_len=len(prop_data_transformed[0])

    whole_m6a_seq=sequences
    i=0
    phisical_chemichy_len=len(prop_data_transformed)#the length of properties
    sequence_line_len=len(sequences[0])#the length of one sequence
    LAMDA=4
    finally_result=[]#used to save the fanal result
    for one_m6a_sequence_line in whole_m6a_seq:
        one_sequence_value=[[]]*(sequence_line_len-1)
        PC_m=[0.0]*prop_len
        PC_m=np.array(PC_m)
        for one_sequence_index in range(sequence_line_len-1):
            for prop_index in list(range(len(prop_key))):
                if one_m6a_sequence_line[one_sequence_index:one_sequence_index+2]==prop_key[prop_index]:
                    one_sequence_value[one_sequence_index]=prop_data_transformed[prop_index]
            PC_m+=np.array(one_sequence_value[one_sequence_index])
        PC_m=PC_m/(sequence_line_len-1)
        auto_value=[]
        for LAMDA_index in list(range(1,LAMDA+1)):
            temp = [0.0] * prop_len
            temp=np.array(temp)
            for auto_index in list(range(1,sequence_line_len-LAMDA_index)):
                temp=temp+(np.array(one_sequence_value[auto_index-1])-PC_m)*(np.array(one_sequence_value[auto_index+LAMDA_index-1])-PC_m)
                temp=[round(e,8) for e in temp.astype(float)]
            x=[round(e/(sequence_line_len-LAMDA_index-1),8) for e in temp]
            auto_value.extend([round(e,8) for e in x])
        for LAMDA_index in list(range(1, LAMDA + 1)):
            for i in list(range(1,prop_len+1)):
                for j in list(range(1,prop_len+1)):
                    temp2=0.0
                    if i != j:
                        for auto_index in list(range(1, sequence_line_len - LAMDA_index)):
                                temp2+=(one_sequence_value[auto_index-1][i-1]-PC_m[i-1])*(one_sequence_value[auto_index+LAMDA_index-1][j-1]-PC_m[j-1])
                        auto_value.append(round(temp2/((sequence_line_len-1)-LAMDA_index),8))

        finally_result.append(auto_value)
    return np.array(finally_result)#.reshape(len(finally_result),-1,1)

def DBPF(sequences):
    DB = []
    for sequence in sequences:
        alphabet="ACGT"
        k_num=2
        two_sequence=[]
        for index,data in enumerate(sequence):
            if index <(len(sequence)-k_num+1):
                two_sequence.append("".join(sequence[index:(index+k_num)]))
        parameter=[e for e in itertools.product([0,1],repeat=4)]
        record=[0 for x in range(int(pow(4,k_num)))]
        matrix=["".join(e) for e in itertools.product(alphabet, repeat=k_num)] # AA AU AC AG UU UC ...
        final=[]
        for index,data in enumerate(two_sequence):
            final_one = []
            if data in matrix:
                final_one.extend(parameter[matrix.index(data)])
                record[matrix.index(data)]+=1
                final_one.append(record[matrix.index(data)]*1.0/(index+1))
            final.append(final_one)
        DB.append(final)
    return np.array(DB)