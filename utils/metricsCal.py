from sklearn.metrics import roc_curve,auc,roc_auc_score
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
# import sys
# sys.path.append("../")
# from Resnet import Res_Net
#import tqdm

#calculate the auc score
def auc_cal(probs, targets):
    if isinstance(probs, torch.Tensor) and isinstance(targets, torch.Tensor):
        fpr, tpr, thresholds = roc_curve(y_true=targets.detach().cpu().numpy(),
                                         y_score=probs.detach().cpu().numpy()) 
    elif isinstance(probs, np.ndarray) and isinstance(targets, np.ndarray):
         fpr, tpr, thresholds = roc_curve(y_true=targets,y_score=probs)
    else:
        print('ERROR: probs or targets type is error.')
        raise TypeError
    auc_ = auc(x=fpr, y=tpr)
    return auc_

#draw the auroc curve
def auc_curve(prob,y):
    fpr, tpr, threshold = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #plt.savefig("BERT.png")
    #plt.show()

#test data and return metrics of results 
def evaluate(model, dataloader, device, is_train=True, threshold=0.5):
    model.eval()
    y_true = torch.tensor([],dtype=torch.int)
    y_score = torch.tensor([])
    #for data in tqdm(dataloader):
    model = model.to(device)
    for data in dataloader:
        #if not isinstance(model, Res_Net):
        if 1==1:
            inputs,y = data
            inputs = inputs.to(device)
            out = model(inputs)
        out = out.squeeze(dim=-1)
        #out = torch.sigmoid(out)
        y_true = torch.cat((y_true, y.int().detach().cpu()))
        y_score = torch.cat((y_score, out.detach().cpu()))
    y_true = y_true.numpy()
    y_score = y_score.numpy()
    if is_train:
        return get_train_metrics(y_score, y_true)
    else:
        return get_test_metrics(y_score, y_true, threshold)

#evaluate model's results
def evaluate_result(model, dataloader, device, is_train=True, threshold=0.5):
    model.eval()
    y_true = torch.tensor([],dtype=torch.int)
    y_score = torch.tensor([])
    #for data in tqdm(dataloader):
    model = model.to(device)
    for data in dataloader:
        #if not isinstance(model, Res_Net):
        if 1==1:
            inputs,y = data
            inputs = inputs.to(device)
            out = model(inputs)
        out = out.squeeze(dim=-1)
        #out = torch.sigmoid(out)
        y_true = torch.cat((y_true, y.int().detach().cpu()))
        y_score = torch.cat((y_score, out.detach().cpu()))
    y_true = y_true.numpy()
    y_score = y_score.numpy()
    return y_score, y_true

#return the attention output and model's results
def evaluate_attn(model, dataloader, device, is_train=True, threshold=0.5):
    model.eval()
    y_true = torch.tensor([],dtype=torch.int)
    y_score = torch.tensor([])
    attn_all = torch.tensor(np.empty (shape= [0,41,41]))
    #for data in tqdm(dataloader):
    model = model.to(device)
    for data in dataloader:
        #if not isinstance(model, Res_Net):
        if 1==1:
            inputs,y = data
            inputs = inputs.to(device)
            out,attn = model(inputs)
        out = out.squeeze(dim=-1)
        #out = torch.sigmoid(out)
        attn_all = torch.cat((attn_all, attn.detach().cpu()))
        y_true = torch.cat((y_true, y.int().detach().cpu()))
        y_score = torch.cat((y_score, out.detach().cpu()))
    y_true = y_true.numpy()
    y_score = y_score.numpy()
    if is_train:
        return attn_all,get_train_metrics(y_score, y_true)
    else:
        return attn_all,get_test_metrics(y_score, y_true, threshold)

#calculate the metrics of the training dataset
def get_train_metrics(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    desc_score_indices = np.argsort(y_pred, kind="mergesort")[::-1]
    y_pred = y_pred[desc_score_indices]
    y_true = y_true[desc_score_indices]

    TP = FP = 0
    TN = np.sum(y_true == 0) 
    FN = np.sum(y_true == 1)  
    mcc = 0
    mcc_threshold = y_pred[0] + 1 
    confuse_matrix = (TP, FP, TN, FN)  
    max_mcc = -1 
    
    for index, score in enumerate(y_pred):  
        if y_true[index] == 1:
            TP += 1
            FN -= 1
        else:
            FP += 1
            TN -= 1
        numerator = (TP * TN - FP * FN)
        denominator = (math.sqrt(TP + FP) * math.sqrt(TN + FN) * math.sqrt(TP + FN) * math.sqrt(TN + FP))
        if denominator == 0:
            mcc = 0
        else:
            mcc = numerator / denominator

        if mcc > max_mcc:
            max_mcc = mcc
            confuse_matrix = (TP, FP, TN, FN)
            mcc_threshold = score 
    TP, FP, TN, FN = confuse_matrix 
    Sen = 0 if (TP + FN) == 0 else (TP / (TP + FN))  
    Spe = 0 if (TN + FP) == 0 else (TN / (TN + FP))
    Acc = 0 if (TP + FP + TN + FN) == 0 else ((TP + TN) / (TP + FP + TN + FN))
    AUC = roc_auc_score(y_true, y_pred)
    return mcc_threshold, TN, FN, FP, TP, Sen, Spe, Acc, max_mcc, AUC 

#calculate the metrics and suitable threshold of test results
def get_test_metrics(y_pred, y_true, threshold):
    # print(threshold)
    # print(y_pred)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    TP = TN = FP = FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] >= threshold:
            TP += 1
        elif y_true[i] == 1 and y_pred[i] < threshold:
            FN += 1
        elif y_true[i] == 0 and y_pred[i] >= threshold:
            FP += 1
        elif y_true[i] == 0 and y_pred[i] < threshold:
            TN += 1
    Sen = 0 if (TP + FN) == 0 else (TP / (TP + FN))
    Spe = 0 if (TN + FP) == 0 else (TN / (TN + FP))
    Acc = 0 if (TP + FP + TN + FN) == 0 else ((TP + TN) / (TP + FP + TN + FN))
    AUC = roc_auc_score(y_true, y_pred)
    #AUC = auc_cal(y_pred,y_true)
    #auc_curve(y_pred,y_true)
    numerator = (TP * TN - FP * FN)
    denominator = (math.sqrt(TP + FP) * math.sqrt(TN + FN) * math.sqrt(TP + FN) * math.sqrt(TN + FP))
    if denominator == 0:
        mcc = 0
    else:
        mcc = numerator / denominator
    return TN, FN, FP, TP, Sen, Spe, Acc, mcc, AUC