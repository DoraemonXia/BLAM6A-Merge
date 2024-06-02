import pandas as pd
import numpy as np
import argparse

#transfer fasta file to csv file
def csv_fa(type_name,cell_name,k_fold):
    seq = list( pd.read_csv("Result/"+type_name+"_"+cell_name+"/KFold_"+str(k_fold)+"/blastn_valid_seq.csv")['seq'] )
    label = list( pd.read_csv("Result/"+type_name+"_"+cell_name+"/KFold_"+str(k_fold)+"/blastn_valid_seq.csv")['label'] )
    
    name_list = []
    for i in range(len(seq)):
        if label[i]==0:
            name_list.append("N"+str(i))
        if label[i]==1:
            name_list.append("P"+str(i))
    
    generate_fasta(seq,name_list,"data/"+type_name+"+"+cell_name+"/Kfold_"+str(k_fold)+".fasta")

#generate the fasta files
def generate_fasta(rna_sequences, names=None, fasta_file_path='output.fasta', reverse=False):
    """
    Generate a FASTA file from RNA sequences.

    Parameters:
    - rna_sequences (list): List of RNA sequences.
    - names (list, optional): List of names corresponding to RNA sequences. If None, default names will be used.
    - fasta_file_path (str, optional): Path to save the generated FASTA file.
    - reverse (bool, optional): If True, replace 'T' with 'U' in the sequences.

    Returns:
    - None
    """
    if not names:
        # If names are not provided, use default names (seq_0, seq_1, ...)
        names = [f"seq_{i}" for i in range(len(rna_sequences))]
    
    if len(rna_sequences) != len(names):
        raise ValueError("Number of RNA sequences must match the number of names.")

    with open(fasta_file_path, 'w') as fasta_file:
        for i in range(len(rna_sequences)):
            sequence = rna_sequences[i]
            name = names[i]
            
            if reverse:
                # If reverse is True, replace 'T' with 'U'
                sequence = sequence.replace('T', 'U')

            fasta_file.write(f">{name}\n{sequence}\n")

#read fasta files
def read_fasta_headers(fasta_file):
    headers = []
    with open(fasta_file, 'r') as file:
        for line in file:
            if line.startswith('>'):
                header = line[1:].strip()
                headers.append(header)
    return headers

#get the results from blast.out file
def get_blastn_results(type_name,cell_name,kfold=-1):
    if kfold==-1:
        fasta_file = "data/"+type_name+"+"+cell_name+"/test.fasta"
        with open("data/"+type_name+"+"+cell_name+"/test.out","r") as file:
            content = file.readlines()

    else:
        fasta_file = "data/"+type_name+"+"+cell_name+"/Kfold_"+str(kfold)+".fasta"
        with open("data/"+type_name+"+"+cell_name+"/KFold_"+str(kfold)+".out","r") as file:
            content = file.readlines()

    headers = read_fasta_headers(fasta_file)
    labels = [1 if 'P' in i else 0 for i in headers]

    blastn_dict = {}
    for i in range(len(headers)):
        if headers[i] not in blastn_dict.keys():
            blastn_dict[headers[i]] = []

    for i in range(len(content)):
        bit_score = float( content[i].split("\t")[-1].split("\n")[0] )
        name = content[i].split("\t")[1]
        blastn_dict[ content[i].split("\t")[0] ].append( (name, bit_score) )
    
    blastn_res = []
    for i in range(len(headers)):
        cont = blastn_dict[headers[i]]
        sum_ = 0
        pos_ = 0
        for j in cont:
            if j[1]!=76.8:
                if 'P' in j[0]:
                    pos_+=j[1]
                sum_+=j[1]
        
        if sum_!=0:
            res = pos_/sum_  
        else:
            res=0.5
        blastn_res.append(res)

    if kfold==-1:
        pd.DataFrame( np.vstack((blastn_res,labels)).T ).to_csv("Result/"+type_name+"_"+cell_name+"/blastn_test_results.csv",header=["pred","label"])
    else:
        pd.DataFrame( np.vstack((blastn_res,labels)).T ).to_csv("Result/"+type_name+"_"+cell_name+"/KFold_"+str(kfold)+"/blastn_valid_results.csv",header=["pred","label"])


if __name__ == '__main__':
    
    #create the argparse
    parser = argparse.ArgumentParser(description='Your script description')

	#Add the parameters
    parser.add_argument('--type_name', type=str, required=True, help='Path to the type data')
    parser.add_argument('--cell_name', type=str, required=True, help='Path to the cell data')
    parser.add_argument('--K_Fold', type=int, required=False, default=-1, help='the num of K_Fold')
    parser.add_argument('--task_name', type=str, required=False, default="generate_fasta", help='If the task is generate_fasta or transfer the .out blastn results')
    #{"generate_fasta","blastn_results"}

    #parser.add_argument('--k_fold', type=str2bool, nargs='?', const=True, default=True, help='Enable or disable BLASTN')
    args = parser.parse_args()

    type_name = args.type_name
    cell_name = args.cell_name
    k_fold = args.K_Fold
    task_name = args.task_name

    if task_name == "generate_fasta":
        if k_fold!=-1:
            csv_fa( type_name, cell_name, k_fold)
        else:
            for i in range(5):
                csv_fa( type_name, cell_name, i )
    elif task_name == "blastn_results":
        if k_fold!=-1:
            get_blastn_results( type_name, cell_name, k_fold )
        for i in range(5):
            get_blastn_results( type_name, cell_name, i )
        get_blastn_results( type_name, cell_name )

