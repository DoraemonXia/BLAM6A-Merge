from sklearn.metrics import roc_curve,auc,roc_auc_score
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
# import sys
# sys.path.append("../")
# from Resnet import Res_Net
#import tqdm

def auc_cal(probs, targets):
    if isinstance(probs, torch.Tensor) and isinstance(targets, torch.Tensor):  #两个都是torch.Tensor
        fpr, tpr, thresholds = roc_curve(y_true=targets.detach().cpu().numpy(),
                                         y_score=probs.detach().cpu().numpy())    #传入两份参数，先标签，再结果，得到fpr,tpr,阈值
    elif isinstance(probs, np.ndarray) and isinstance(targets, np.ndarray):  #两个都是np.ndarray
         fpr, tpr, thresholds = roc_curve(y_true=targets,y_score=probs)
    else:
        print('ERROR: probs or targets type is error.')
        raise TypeError
    auc_ = auc(x=fpr, y=tpr)
    return auc_


def auc_curve(prob,y):
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #plt.savefig("BERT.png")
    #plt.show()

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
            #inputs = inputs.to(device)
            inputs_emb = inputs[:,:,:3800].reshape(inputs.shape[0],38,100)
            inputs_PSNP = inputs[:,:,3800:3841].reshape(inputs.shape[0],1,41)
            inputs_PCP = inputs[:,:,3841:4241].reshape(inputs.shape[0],1,400)
            inputs_DBPF = inputs[:,:,4241:4441].reshape(inputs.shape[0],40,5)
            inputs_emb = inputs_emb.to(device)
            inputs_PSNP = inputs_PSNP.to(device)
            inputs_PCP = inputs_PCP.to(device)
            inputs_DBPF = inputs_DBPF.to(device)
            #out = model(inputs)
            out = model(inputs_emb,inputs_PSNP,inputs_PCP,inputs_DBPF)
        out = out.squeeze(dim=-1)
        #out = torch.sigmoid(out)
        y_true = torch.cat((y_true, y.int().detach().cpu()))
        y_score = torch.cat((y_score, out.detach().cpu()))  #detach去除梯度，然后cpu()，然后cat将其连接起来
    y_true = y_true.numpy()
    y_score = y_score.numpy()
    if is_train:
        return get_train_metrics(y_score, y_true)
    else:
        return get_test_metrics(y_score, y_true, threshold)

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
            #inputs = inputs.to(device)
            inputs_emb = inputs[:,:,:3800].reshape(inputs.shape[0],38,100)
            inputs_PSNP = inputs[:,:,3800:3841].reshape(inputs.shape[0],1,41)
            inputs_PCP = inputs[:,:,3841:4241].reshape(inputs.shape[0],1,400)
            inputs_DBPF = inputs[:,:,4241:4441].reshape(inputs.shape[0],40,5)
            inputs_emb = inputs_emb.to(train_device)
            inputs_PSNP = inputs_PSNP.to(train_device)
            inputs_PCP = inputs_PCP.to(train_device)
            inputs_DBPF = inputs_DBPF.to(train_device)
            #out = model(inputs)
            out = model(inputs_emb,inputs_PSNP,inputs_PCP,inputs_DBPF)
        out = out.squeeze(dim=-1)
        #out = torch.sigmoid(out)
        y_true = torch.cat((y_true, y.int().detach().cpu()))
        y_score = torch.cat((y_score, out.detach().cpu()))  #detach去除梯度，然后cpu()，然后cat将其连接起来
    y_true = y_true.numpy()
    y_score = y_score.numpy()
    return y_score, y_true
    
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
        y_score = torch.cat((y_score, out.detach().cpu()))  #detach去除梯度，然后cpu()，然后cat将其连接起来
    y_true = y_true.numpy()
    y_score = y_score.numpy()
    if is_train:
        return attn_all,get_train_metrics(y_score, y_true)
    else:
        return attn_all,get_test_metrics(y_score, y_true, threshold)
    
def get_train_metrics(y_pred, y_true):  # 获取训练指标
    y_pred = np.array(y_pred)  # 预测值，转化为numpy
    y_true = np.array(y_true)  # 准确值，转化为numpy
    ## [::-1]反序
    desc_score_indices = np.argsort(y_pred, kind="mergesort")[::-1]  # 使用归并排序，让y_pred降序，返回标签序
    y_pred = y_pred[desc_score_indices]
    y_true = y_true[desc_score_indices]

    TP = FP = 0
    TN = np.sum(y_true == 0)  # 结果为0的数量
    FN = np.sum(y_true == 1)  # 结果为1的数量
    mcc = 0
    mcc_threshold = y_pred[0] + 1  # 预测的最大值加1
    confuse_matrix = (TP, FP, TN, FN)  # 定义混淆矩阵
    max_mcc = -1  # 定义最大mcc值用于记录
    ## score 降序排序， y >= score 前半部分预测为正
    for index, score in enumerate(y_pred):  # 取索引+预测值
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
            mcc_threshold = score  # 找出最好的阈值
    TP, FP, TN, FN = confuse_matrix  # 混淆矩阵得出结果
    Sen = 0 if (TP + FN) == 0 else (TP / (TP + FN))  # 计算……
    Spe = 0 if (TN + FP) == 0 else (TN / (TN + FP))
    Acc = 0 if (TP + FP + TN + FN) == 0 else ((TP + TN) / (TP + FP + TN + FN))
    AUC = roc_auc_score(y_true, y_pred)
    return mcc_threshold, TN, FN, FP, TP, Sen, Spe, Acc, max_mcc, AUC  # 返回一系列结果


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