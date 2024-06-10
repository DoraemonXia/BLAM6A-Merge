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

def str2bool(str):
	return True if str.lower() == 'true' else False

if __name__ == '__main__':
    
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")

    #create the argparse
    parser = argparse.ArgumentParser(description='Your script description')

	#Add the parameters
    parser.add_argument('--type_name', type=str, required=True, help='Path to the type data')
    parser.add_argument('--cell_name', type=str, required=True, help='Path to the cell data')
    parser.add_argument('--if_blastn', type=str2bool, nargs='?', const=True, default=True, help='Enable or disable BLASTN')
    args = parser.parse_args()

    type_name = args.type_name
    cell_name = args.cell_name
    if_blastn = args.if_blastn

    #type_name = ["FullTranscript","matureRNA"]
    #cell_name = ["A549","CD8T","Hek293_abacm","Hek293_sysy","HeLa","MOLM13"]
    #feature_name = ["binary","DNC","NCPA","emb","PSNP","ENAC","EIIP","PseDNC","PCP","DBPF"]

    # Hyper Parameters
    batch_size = 32

    #choose the device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

    # trainPosSeq_file = "data/"+type_name+"/"+cell_name+"/Pos_"+cell_name+"_train_"+type_name[typei]+"_seq.csv"
    # trainNegSeq_file = "data/"+type_name+"/"+cell_name+"/Neg_"+cell_name+"_train_"+type_name[typei]+"_seq.csv"
    # testPosSeq_file = "data/"+type_name+"/"+cell_name+"/Pos_"+cell_name+"_test_"+type_name[typei]+"_seq.csv"
    # testNegSeq_file = "data/"+type_name+"/"+cell_name+"/Neg_"+cell_name+"_test_"+type_name[typei]+"_seq.csv"
    
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
    # trainPos_emb = ensembleFeature.emb_seqs(trainPos_seq)
    # trainNeg_emb = ensembleFeature.emb_seqs(trainNeg_seq)
    testPos_emb = ensembleFeature.emb_seqs(testPos_seq)
    testNeg_emb = ensembleFeature.emb_seqs(testNeg_seq)

    print("The model is generating the PSNP feature from the seq now!")
    trainPos_PSNP,trainNeg_PSNP,testPos_PSNP,testNeg_PSNP = ensembleFeature.PSNP(trainPos_seq,trainNeg_seq,testPos_seq,testNeg_seq)

    print("The model is generating the PCP feature from the seq now, please waiting……,this will spend some time")
    # trainPos_PCP = ensembleFeature.PCP(trainPos_seq)
    # trainNeg_PCP = ensembleFeature.PCP(trainNeg_seq)
    testPos_PCP = ensembleFeature.PCP(testPos_seq)
    testNeg_PCP = ensembleFeature.PCP(testNeg_seq)

    print("The model is generating the DBPF feature from the seq now!")
    # trainPos_DBPF = ensembleFeature.DBPF(trainPos_seq)
    # trainNeg_DBPF = ensembleFeature.DBPF(trainNeg_seq)
    testPos_DBPF = ensembleFeature.DBPF(testPos_seq)
    testNeg_DBPF = ensembleFeature.DBPF(testNeg_seq)


    # trainPos_PSNP = trainPos_PSNP.reshape(trainPos_PSNP.shape[0],1,trainPos_PSNP.shape[1])
    # trainNeg_PSNP = trainNeg_PSNP.reshape(trainNeg_PSNP.shape[0],1,trainNeg_PSNP.shape[1])
    testPos_PSNP = testPos_PSNP.reshape(testPos_PSNP.shape[0],1,testPos_PSNP.shape[1])
    testNeg_PSNP = testNeg_PSNP.reshape(testNeg_PSNP.shape[0],1,testNeg_PSNP.shape[1])
    
    # trainPos_PCP = trainPos_PCP.reshape(trainPos_PCP.shape[0],1,trainPos_PCP.shape[1])
    # trainNeg_PCP = trainNeg_PCP.reshape(trainNeg_PCP.shape[0],1,trainNeg_PCP.shape[1])
    testPos_PCP = testPos_PCP.reshape(testPos_PCP.shape[0],1,testPos_PCP.shape[1])
    testNeg_PCP = testNeg_PCP.reshape(testNeg_PCP.shape[0],1,testNeg_PCP.shape[1])

    testData_emb = np.append(testPos_emb,testNeg_emb,axis = 0)
    testData_PSNP = np.append(testPos_PSNP,testNeg_PSNP,axis = 0)
    testData_DBPF = np.append(testPos_DBPF,testNeg_DBPF,axis = 0)
    testData_PCP = np.append(testPos_PCP,testNeg_PCP,axis = 0)

    testLabel = np.append(np.ones(len(testPos_PCP)),np.zeros(len(testNeg_PCP)),axis = 0)


    #Do the independent test for emb
    for i in range(5):
        print("Fold_",str(i))
        print("get the results from model")
        Output_path = 'Result/'+type_name+'_'+cell_name+'/'

        if not os.path.exists(Output_path+'KFold_'+str(i)):
            os.makedirs(Output_path+'KFold_'+str(i))
        
        #test the model for w2v feature
        model_dir = "Model/"+type_name+"_"+cell_name+"/emb/KFold_" + str(i) + "/"
        model = Model.ModelB_MultiSelf_Pro(testData_emb.shape[1],testData_emb.shape[2])
        y_pred_emb,y_true = Model.independResult(testData_emb,testLabel,device,model_dir,64,0.5)
        
        #test the model for PCP feature
        model_dir = "Model/"+type_name+"_"+cell_name+"/PCP/KFold_" + str(i) + "/"
        model = Model.ModelBS_Pro(testData_PCP.shape[1],testData_PCP.shape[2])
        y_pred_PCP,y_true = Model.independResult(testData_PCP,testLabel,device,model_dir,64,0.5)

        #test the model for PSNP feature
        model_dir = "Model/"+type_name+"_"+cell_name+"/PSNP/KFold_" + str(i) + "/"
        model = Model.ModelB_MultiSelf_Pro(testData_PSNP.shape[1],testData_PSNP.shape[2])
        y_pred_PSNP,y_true = Model.independResult(testData_PSNP,testLabel,device,model_dir,64,0.5)

        #test the model for DBPF feature
        model_dir = "Model/"+type_name+"_"+cell_name+"/DBPF/KFold_" + str(i) + "/"
        model = Model.ModelB_Bah_Pro(testData_DBPF.shape[1],testData_DBPF.shape[2])
        y_pred_DBPF,y_true = Model.independResult(testData_DBPF,testLabel,device,model_dir,64,0.5)

        if args.if_blastn:
            # If you don't do Blastn, you can use the data once i get from Blastn.
            blastn_res = list( pd.read_csv('Blastn/'+type_name+'_'+cell_name+'_result.csv',header=None,index_col=0).iloc[:,0])
            
            # You can also use your's Blastn Results like this.
            # blastn_res = list( pd.read_csv('Result/'+type_name+'_'+cell_name+'/blastn_test_results.csv',header=None,index_col=0).iloc[:,0])
            
            df = pd.DataFrame( np.vstack((testLabel,y_pred_emb,y_pred_PCP,y_pred_PSNP,y_pred_DBPF,blastn_res)).T)
            df.to_csv('Result/'+type_name+'_'+cell_name+'/test_multi_results.csv',columns = ["label","w2v","PCP","PSNP","DBPF","blastn"], header=True,index=False)
        else:
            df = pd.DataFrame( np.vstack((testLabel,y_pred_emb,y_pred_PCP,y_pred_PSNP,y_pred_DBPF)).T)
            df.to_csv('Result/'+type_name+'_'+cell_name+'/KFold_'+str(i)+'/test_multi_results.csv',columns = ["label","w2v","PCP","PSNP","DBPF"], header=True,index=False)
        
        print('analysis the results.')

        #Get the validation results.
        valid_results = pd.read_csv('Result/'+type_name+'_'+cell_name+'/KFold_'+str(i)+'/valid_results.csv',header=None)
        if args.if_blastn:
            blastn_res = list( pd.read_csv('Result/'+type_name+'_'+cell_name+'/KFold_'+str(i)+'/blastn_valid_results.csv').iloc[:,0])
        stack_train = np.array( np.vstack((valid_results.iloc[:,1:],blastn_res )).T )
        stack_label = np.array( valid_results.iloc[:,0] )

        #This is the final Results.
        new_pred, TN, FN, FP, TP, Sen, Spe, Acc, mcc, AUC = Model.analysis_results( np.array(df.iloc[:,1:]),y_true, stack_train, stack_label, strategy="stack")
        print(type_name," ",cell_name," :")
        df = pd.DataFrame(np.vstack((new_pred,y_true)).T)
        df.columns=["pred","label"]
        df.to_csv('Result/'+type_name+'_'+cell_name+'/KFold_'+str(i)+'/final_results.csv', header=True,index=False)

        print('Accuracy on test set: ', Acc)
        print('Sensitivity on test set: ', Sen)
        print('Speciality on test set: ', Spe)
        print('MCC on test set: %.3f' %mcc)
        print('auc on test set: %.3f' %AUC)