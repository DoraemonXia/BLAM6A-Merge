from features import ensembleFeature
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch
from torch import nn
import re
from sklearn.utils import shuffle
from utils import metricsCal
from torch.utils.data import DataLoader,TensorDataset
import math
import sys
import copy
import pickle
from torch.autograd import Variable
from sklearn.model_selection import KFold
import torch.nn.functional as F
import os
import Attention_model as Model
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import csv
import argparse
# %matplotlib inline

def read_fasta(fasta_file_path):
    """
    Read sequences and names from a FASTA file.

    Parameters:
    - fasta_file_path (str): Path to the FASTA file.

    Returns:
    - sequences (list): List of RNA sequences.
    - names (list): List of names corresponding to RNA sequences.
    """
    sequences = []
    names = []

    with open(fasta_file_path, 'r') as fasta_file:
        for line in fasta_file:
            if line.startswith('>'):
                name = line.strip()[1:]
                names.append(name)
                sequence = ''
            else:
                sequence += line.strip()
                sequences.append(sequence)
    return sequences, names

#change the length of seq from 1001 to 41.
def long_short(data):
    seq_list = []
    for i in data[0]:
        seq_list.append(str(i)[480:521])
    return np.array(seq_list)

# delete the seq if its' bases has 'N', either not the 'ATCG'
def check_N(data1):
    seq_list = []
    for i in range(len(data1)):
        if str(data1[i]).find("N")<0:
            seq_list.append(data1[i])
    return np.array(seq_list)

if __name__ == '__main__':
    
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")

    #create the argparse
    parser = argparse.ArgumentParser(description='Your script description')

	#Add the parameters
    parser.add_argument('--type_name', type=str, required=True, help='Path to the positive training data')
    parser.add_argument('--cell_name', type=str, required=True, help='Path to the negative training data')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--max_patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda_device', type=int, default=0)

    args = parser.parse_args()

    type_name = args.type_name
    cell_name = args.cell_name

    #type_name = ["FullTranscript","matureRNA"]
    #cell_name = ["A549","CD8T","Hek293_abacm","Hek293_sysy","HeLa","MOLM13"]
    #feature_name = ["binary","DNC","NCPA","emb","PSNP","ENAC","EIIP","PseDNC","PCP","DBPF"]

    # Hyper Parameters
    max_epochs = args.max_epochs
    max_patience = args.max_patience
    batch_size = args.batch_size
    cuda_device = args.cuda_device

    #choose the device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_seq,name = read_fasta("data/"+str(type_name)+"+"+str(cell_name)+"/train.fa")

    trainPos_seq = []
    trainNeg_seq = []
    for i in range(len(name)):
        if name[i][0]=='P':
            trainPos_seq.append(train_seq[i])
        else:
            trainNeg_seq.append(train_seq[i])

    test_seq,name = read_fasta("data/"+str(type_name)+"+"+str(cell_name)+"/test.fa")

    testPos_seq = []
    testNeg_seq = []
    for i in range(len(name)):
        if name[i][0]=='P':
            testPos_seq.append(test_seq[i])
        else:
            testNeg_seq.append(test_seq[i])

    # trainPosSeq_file = "data/"+type_name+"_"+cell_name+"/Pos_"+cell_name+"_train_"+type_name[typei]+"_seq.csv"
    # trainNegSeq_file = "data/"+type_name+"_"+cell_name+"/Neg_"+cell_name+"_train_"+type_name[typei]+"_seq.csv"
    # testPosSeq_file = "data/"+type_name+"_"+cell_name+"/Pos_"+cell_name+"_test_"+type_name[typei]+"_seq.csv"
    # testNegSeq_file = "data/"+type_name+"_"+cell_name+"/Neg_"+cell_name+"_test_"+type_name[typei]+"_seq.csv"
    
    # #get the file information
    # trainPos_seq = pd.read_csv(trainPosSeq_file,header=None)
    # trainNeg_seq = pd.read_csv(trainNegSeq_file,header=None)
    # testPos_seq = pd.read_csv(testPosSeq_file,header=None)
    # testNeg_seq = pd.read_csv(testNegSeq_file,header=None)

    # #change the length to 41
    # trainPos_seq = long_short(trainPos_seq)
    # trainNeg_seq = long_short(trainNeg_seq)
    # testPos_seq = long_short(testPos_seq)
    # testNeg_seq = long_short(testNeg_seq)
    
    # #delete the seq which included 'N' base
    # trainPos_seq = check_N(trainPos_seq)
    # trainNeg_seq = check_N(trainNeg_seq)
    # testPos_seq = check_N(testPos_seq)
    # testNeg_seq = check_N(testNeg_seq)

    #Shuffle the seq
    trainNeg_seq = shuffle(trainNeg_seq, random_state=1)
    testNeg_seq = shuffle(testNeg_seq, random_state=1)

    #choose the seq
    trainNeg_seq = trainNeg_seq[:len(trainPos_seq)*10]
    testNeg_seq = testNeg_seq[:len(testPos_seq)]

    #Generate the feature.
    print("The model is generating the w2v feature from the seq now!")
    trainPos_emb = ensembleFeature.emb_seqs(trainPos_seq)
    trainNeg_emb = ensembleFeature.emb_seqs(trainNeg_seq)
    # testPos_emb = ensembleFeature.emb_seqs(testPos_seq)
    # testNeg_emb = ensembleFeature.emb_seqs(testNeg_seq)

    print("The model is generating the PSNP feature from the seq now!")
    trainPos_PSNP,trainNeg_PSNP,testPos_PSNP,testNeg_PSNP = ensembleFeature.PSNP(trainPos_seq,trainNeg_seq,testPos_seq,testNeg_seq)

    print("The model is generating the PCP feature from the seq now, please waiting……,this will spend some time")
    trainPos_PCP = ensembleFeature.PCP(trainPos_seq)
    trainNeg_PCP = ensembleFeature.PCP(trainNeg_seq)
    # testPos_PCP = ensembleFeature.PCP(testPos_seq)
    # testNeg_PCP = ensembleFeature.PCP(testNeg_seq)

    print("The model is generating the DBPF feature from the seq now!")
    trainPos_DBPF = ensembleFeature.DBPF(trainPos_seq)
    trainNeg_DBPF = ensembleFeature.DBPF(trainNeg_seq)
    # testPos_DBPF = ensembleFeature.DBPF(testPos_seq)
    # testNeg_DBPF = ensembleFeature.DBPF(testNeg_seq)


    trainPos_PSNP = trainPos_PSNP.reshape(trainPos_PSNP.shape[0],1,trainPos_PSNP.shape[1])
    trainNeg_PSNP = trainNeg_PSNP.reshape(trainNeg_PSNP.shape[0],1,trainNeg_PSNP.shape[1])
    # testPos_PSNP = testPos_PSNP.reshape(testPos_PSNP.shape[0],1,testPos_PSNP.shape[1])
    # testNeg_PSNP = testNeg_PSNP.reshape(testNeg_PSNP.shape[0],1,testNeg_PSNP.shape[1])
    
    trainPos_PCP = trainPos_PCP.reshape(trainPos_PCP.shape[0],1,trainPos_PCP.shape[1])
    trainNeg_PCP = trainNeg_PCP.reshape(trainNeg_PCP.shape[0],1,trainNeg_PCP.shape[1])
    # testPos_PCP = testPos_PCP.reshape(testPos_PCP.shape[0],1,testPos_PCP.shape[1])
    # testNeg_PCP = testNeg_PCP.reshape(testNeg_PCP.shape[0],1,testNeg_PCP.shape[1])

    #Do the 5-fold cross validation
    
    kf = KFold(5,shuffle=True)#,True)#,10)
    for i,[train_index, test_index] in enumerate(kf.split(trainNeg_emb)):
    
    #generate the data for training and validation.
        X_train_PCP_neg = trainNeg_PCP[train_index]
        X_test_PCP_neg = trainNeg_PCP[test_index]
        trainPos_PCP = np.repeat(trainPos_PCP,10,axis=0)
        X_train_PCP_pos = trainPos_PCP[:int(0.8*len(trainPos_PCP))]
        X_test_PCP_pos = trainPos_PCP[int(0.8*len(trainPos_PCP)):]
        Y_train_PCP = np.append(np.ones(len(X_train_PCP_pos)),np.zeros(len(X_train_PCP_neg)),axis = 0)
        Y_test_PCP = np.append(np.ones(len(X_test_PCP_pos)),np.zeros(len(X_test_PCP_neg)),axis = 0)
        X_train_PCP = np.append(X_train_PCP_pos,X_train_PCP_neg,axis=0)
        X_test_PCP = np.append(X_test_PCP_pos,X_test_PCP_neg,axis=0)
        X_train_PCP, Y_train_PCP = shuffle(X_train_PCP, Y_train_PCP,random_state=42)
        X_test_PCP, Y_test_PCP = shuffle(X_test_PCP, Y_test_PCP,random_state=42)

        X_train_emb_neg = trainNeg_emb[train_index]
        X_test_emb_neg = trainNeg_emb[test_index]
        trainPos_emb = np.repeat(trainPos_emb,10,axis=0)
        X_train_emb_pos = trainPos_emb[:int(0.8*len(trainPos_emb))]
        X_test_emb_pos = trainPos_emb[int(0.8*len(trainPos_emb)):]
        Y_train_emb = np.append(np.ones(len(X_train_emb_pos)),np.zeros(len(X_train_emb_neg)),axis = 0)
        Y_test_emb = np.append(np.ones(len(X_test_emb_pos)),np.zeros(len(X_test_emb_neg)),axis = 0)
        X_train_emb = np.append(X_train_emb_pos,X_train_emb_neg,axis=0)
        X_test_emb = np.append(X_test_emb_pos,X_test_emb_neg,axis=0)
        X_train_emb, Y_train_emb = shuffle(X_train_emb, Y_train_emb,random_state=42)
        X_test_emb, Y_test_emb = shuffle(X_test_emb, Y_test_emb,random_state=42)

        X_train_DBPF_neg = trainNeg_DBPF[train_index]
        X_test_DBPF_neg = trainNeg_DBPF[test_index]
        trainPos_DBPF = np.repeat(trainPos_DBPF,10,axis=0)
        X_train_DBPF_pos = trainPos_DBPF[:int(0.8*len(trainPos_DBPF))]
        X_test_DBPF_pos = trainPos_DBPF[int(0.8*len(trainPos_DBPF)):]
        Y_train_DBPF = np.append(np.ones(len(X_train_DBPF_pos)),np.zeros(len(X_train_DBPF_neg)),axis = 0)
        Y_test_DBPF = np.append(np.ones(len(X_test_DBPF_pos)),np.zeros(len(X_test_DBPF_neg)),axis = 0)
        X_train_DBPF = np.append(X_train_DBPF_pos,X_train_DBPF_neg,axis=0)
        X_test_DBPF = np.append(X_test_DBPF_pos,X_test_DBPF_neg,axis=0)
        X_train_DBPF, Y_train_DBPF = shuffle(X_train_DBPF, Y_train_DBPF,random_state=42)
        X_test_DBPF, Y_test_DBPF = shuffle(X_test_DBPF, Y_test_DBPF,random_state=42)

        X_train_PSNP_neg = trainNeg_PSNP[train_index]
        X_test_PSNP_neg = trainNeg_PSNP[test_index]
        trainPos_PSNP = np.repeat(trainPos_PSNP,10,axis=0)
        X_train_PSNP_pos = trainPos_PSNP[:int(0.8*len(trainPos_PSNP))]
        X_test_PSNP_pos = trainPos_PSNP[int(0.8*len(trainPos_PSNP)):]
        Y_train_PSNP = np.append(np.ones(len(X_train_PSNP_pos)),np.zeros(len(X_train_PSNP_neg)),axis = 0)
        Y_test_PSNP = np.append(np.ones(len(X_test_PSNP_pos)),np.zeros(len(X_test_PSNP_neg)),axis = 0)
        X_train_PSNP = np.append(X_train_PSNP_pos,X_train_PSNP_neg,axis=0)
        X_test_PSNP = np.append(X_test_PSNP_pos,X_test_PSNP_neg,axis=0)
        X_train_PSNP, Y_train_PSNP = shuffle(X_train_PSNP, Y_train_PSNP,random_state=42)
        X_test_PSNP, Y_test_PSNP = shuffle(X_test_PSNP, Y_test_PSNP,random_state=42)

        #train the model for emb feature
        best_auc = 0
        patience = 0
        model_dir = "Model/"+type_name+"_"+cell_name+"/emb/KFold_" + str(i) + "/"
        
        model = Model.ModelB_MultiSelf_Pro(X_train_emb.shape[1],X_test_emb.shape[2])
        
        for j in range(max_epochs):
            runningLoss,th = Model.train(model,X_train_emb,Y_train_emb,i,device,model_dir,batch_size)
            acc, mcc, auc=Model.test(X_test_emb,Y_test_emb,best_auc,device,model_dir,batch_size,th)

            if auc > best_auc:
                best_auc=auc
                patience=0
            else:
                patience+=1
            if patience==max_patience:
                break
        y_pred_emb, y_true = Model.independResult( X_test_emb,Y_test_emb,device,model_dir,batch_size,th )
        
        print("The model for emb feature had trained over!")

        #train the model for PCP feature
        best_auc = 0
        patience = 0
        model_dir = "Model/"+type_name+"_"+cell_name+"/PCP/KFold_" + str(i) + "/"
        
        model = Model.ModelBS_Pro(X_train_PCP.shape[1],X_test_PCP.shape[2])
        
        for j in range(max_epochs):
            runningLoss,th = Model.train(model,X_train_PCP,Y_train_PCP,i,device,model_dir,batch_size)
            acc, mcc, auc=Model.test(X_test_PCP,Y_test_PCP,best_auc,device,model_dir,batch_size,th)

            if auc > best_auc:
                best_auc=auc
                patience=0
            else:
                patience+=1
            if patience==max_patience:
                break
        y_pred_PCP, y_true = Model.independResult( X_test_PCP,Y_test_PCP,device,model_dir,batch_size,th )
        print("The model for PCP feature had trained over!")

        #train the model for PSNP feature
        best_auc = 0
        patience = 0
        model_dir = "Model/"+type_name+"_"+cell_name+"/PSNP/KFold_" + str(i) + "/"
        
        model = Model.ModelB_MultiSelf_Pro(X_train_PSNP.shape[1],X_test_PSNP.shape[2])
        
        for j in range(max_epochs):
            runningLoss,th = Model.train(model,X_train_PSNP,Y_train_PSNP,i,device,model_dir,batch_size)
            acc, mcc, auc=Model.test(X_test_PSNP,Y_test_PSNP,best_auc,device,model_dir,batch_size,th)
            print()

            if auc > best_auc:
                best_auc=auc
                patience=0
            else:
                patience+=1
            if patience==max_patience:
                break
        y_pred_PSNP, y_true = Model.independResult( X_test_PSNP,Y_test_PSNP,device,model_dir,batch_size,th )
        print("The model for PSNP feature had trained over!")

        #train the model for DBPF feature
        best_auc = 0
        patience = 0
        model_dir = "Model/"+type_name+"_"+cell_name+"/DBPF/KFold_" + str(i) + "/"
        
        model = Model.ModelB_Bah_Pro(X_train_DBPF.shape[1],X_test_DBPF.shape[2])
        
        for j in range(max_epochs):
            runningLoss,th = Model.train(model,X_train_DBPF,Y_train_DBPF,i,device,model_dir,batch_size)
            acc, mcc, auc=Model.test(X_test_DBPF,Y_test_DBPF,best_auc,device,model_dir,batch_size,th)
            print()

            if auc > best_auc:
                best_auc=auc
                patience=0
            else:
                patience+=1
            if patience==max_patience:
                break
        y_pred_DBPF, y_true = Model.independResult( X_test_DBPF,Y_test_DBPF,device,model_dir,batch_size,th )
        print("The model for DBPF feature had trained over!")

        Output_path = 'Result/'+type_name+'_'+cell_name+'/'
        if not os.path.exists(Output_path+'KFold_'+str(i)):
            os.makedirs(Output_path+'KFold_'+str(i))

        #this will generate the seq for blastn validation.
        trainNeg_seq = np.array(trainNeg_seq)
        trainPos_seq_10 = np.repeat( trainPos_seq,10)
        Blastn_seq = np.append( trainPos_seq_10[int(0.8*len(trainPos_seq_10)):],trainNeg_seq[test_index],axis=0 )
        valid_label = np.append( np.ones( len(trainPos_seq_10[int(0.8*len(trainPos_seq_10)):])), np.zeros(len(trainNeg_seq[test_index]) ),axis=0)
        
        Blastn_seq,valid_label = shuffle( Blastn_seq,valid_label,random_state=42 )
        df = pd.DataFrame( np.vstack((Blastn_seq,valid_label)).T)
        df.to_csv('Result/'+type_name+'_'+cell_name+'/KFold_'+str(i)+'/blastn_valid_seq.csv',header = ["seq","label"],index=False)

        #this will generate the seq for blastn test.
        Blastn_test_seq = np.append(testPos_seq,testNeg_seq,axis=0)
        testLabel = np.append(np.ones(len(testPos_PSNP)),np.zeros(len(testNeg_PSNP)),axis = 0)
        df = pd.DataFrame( np.vstack((Blastn_test_seq,testLabel)).T)
        df.to_csv('Result/'+type_name+'_'+cell_name+'/KFold_'+str(i)+'/blastn_test_seq.csv',header = ["seq","label"],index=False)

        #this is the validation result of different encoding schemes in corresponding KFold.
        df = pd.DataFrame( np.vstack((y_true,y_pred_emb,y_pred_PCP,y_pred_PSNP,y_pred_DBPF)).T)
        df.to_csv('Result/'+type_name+'_'+cell_name+'/KFold_'+str(i)+'/valid_results.csv',header = ["label","w2v","PCP","PSNP","DBPF"],index=False)