import os
import sys
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential,Conv1d,MaxPool1d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
from termcolor import colored
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import pickle
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import pickle
from sklearn import metrics
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
from sklearn.model_selection import KFold
warnings.filterwarnings( "ignore" )
from sklearn.model_selection import ParameterGrid

import joblib
with open('/home/jwen/提取好的特征/twostr-features.pkl', 'rb') as f:
    feature_struct = pickle.load(f)
feature_struct_train=feature_struct[:8038]
feature_struct_test=feature_struct[8038:]
with open('/home/jwen/提取好的特征/featuers_embedding_normalize.pkl', 'rb') as f:
    features_ensemble_train = pickle.load(f)
with open('/home/jwen/提取好的特征/train_handfeature8038gai.pkl', 'rb') as f:
        feature_hand = pickle.load(f)
features_ensemble = np.concatenate((features_ensemble_train, feature_hand), axis=1)
features_ensemble = np.concatenate((features_ensemble,feature_struct_train), axis=1)
train_x = torch.from_numpy(features_ensemble)
print(train_x.shape)
with open('/home/jwen/提取好的特征/test-featuers_embedding_normalize.pkl', 'rb') as f:
    test_features_ensemble = pickle.load(f)
with open('/home/jwen/提取好的特征/test_handfeature8038gai.pkl', 'rb') as f:
    test_feature_hand = pickle.load(f)
test_features = np.concatenate((test_features_ensemble, test_feature_hand), axis=1)
test_features = np.concatenate((test_features, feature_struct_test), axis=1)
test_x = torch.from_numpy(test_features)
print(test_x.shape)

data1 = np.load('/home/jwen/提取好的特征/1mer.npz')
a1 = data1['x_train']
b1 = data1['x_test']

data2 = np.load('/home/jwen/提取好的特征/4mer.npz')
a2 = data2['x_train']
b2 = data2['x_test']
#在水平方向上合并两个矩阵

X11 = np.hstack((a1, a2))
X22 = np.hstack((b1, b2))
train_word = torch.tensor(X11)

test_word = torch.tensor(X22)






def greater_than_half(lst):
    result = []
    for item in lst:
        if item >= 0.5:
            result.append(1)
        else:
            result.append(0)
    return result

def generate_data1(file):
    with open(file, 'r') as inf:
        lines = inf.read().splitlines()
    labels = []
    for pep in lines:
        pep, label = pep.split(",")
        labels.append(int(label))
    return torch.tensor(labels)

def generate_pep(file):
    # Amino acid dictionary
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22, 'X': 23}

    # Secondary structure dictionary
    ss_dict = {'C': 1, 'H': 2, 'E': 3}

    with open(file, 'r') as inf:
        lines = inf.read().splitlines()

    pep_codes = []
    labels = []
    peps = []
    secondary_strus_codes = []
    for pep in lines:
        pep, label = pep.split(",")
        peps.append(pep)
        labels.append(int(label))
        current_pep = []
        for aa in pep:
            current_pep.append(aa_dict[aa])
        pep_codes.append(torch.tensor(current_pep))


    data = rnn_utils.pad_sequence(pep_codes, batch_first=True)  # Fill the sequence to the same length


    return data

label = generate_data1("/home/jwen/提取好的特征/data2.csv")
pep=generate_pep("/home/jwen/提取好的特征/data2.csv")
train_pep=pep[:8038]
test_pep=pep[8038:]

train_y=label[:8038]
test_y=label[8038:]
batch_size=256
test_batch=256
print(train_word.shape)
train_data = Data.TensorDataset(train_x, train_pep,train_word,train_y)
train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = Data.TensorDataset(test_x,test_pep,test_word ,test_y)
test_iter= torch.utils.data.DataLoader(test_data, batch_size=test_batch, shuffle=True)

import torch
import numpy as np
import random
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
# setup_seed(20)
setup_seed(1)



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,a,b,c,d,e):
        super(LSTMModel, self).__init__()
        #transfomer
        self.num_heads=a
        self.num_layer=b
        #cnn
        self.cnn_dropout=c
        #gru
        self.gru_yayer=d
        self.gru_dropout=e

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = 512
        self.max_pool=2
        self.dropout=0.7
        self.liner=nn.Linear(3330,3330)
        self.liner1 = nn.Linear(1282, 1024)
        self.bath1 = nn.BatchNorm1d(256)
        self.bath2 = nn.BatchNorm1d(256)
        self.bath3 = nn.BatchNorm1d(256)
        self.lstm = nn.LSTM(3330, hidden_size, num_layers, batch_first=True,dropout=0.1)
        self.lstm1 = nn.LSTM(1024, 256, num_layers, batch_first=True, dropout=0.1)
        self.lstm2 = nn.LSTM(1282, 256, num_layers, batch_first=True, dropout=0.1)
        self.lstm3 = nn.LSTM(1024, 256, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(768, output_size)
        self.sigmoid = nn.Sigmoid()
        self.liner2=nn.Flatten()
        self.attention1 = nn.MultiheadAttention(num_heads=1,dropout=0.1,embed_dim=1024)
        self.attention2 = nn.MultiheadAttention(num_heads=1, dropout=0.1, embed_dim=1282)
        self.attention3 = nn.MultiheadAttention(num_heads=1, dropout=0.1, embed_dim=1024)

        self.embedding = nn.Embedding(24, 512, padding_idx=0)#1
        self.encoder_layer_seq = nn.TransformerEncoderLayer(d_model=512, nhead=self.num_heads)#2 a
        self.transformer_encoder_seq = nn.TransformerEncoder(self.encoder_layer_seq, num_layers=self.num_layer)#3b
        self.gru = nn.GRU(192, 25, num_layers=self.gru_yayer, bidirectional=True, dropout=self.gru_dropout)#d e
        self.fc1 = nn.Sequential(nn.Linear(3200, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(),
                                    nn.Linear(1024, 256),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(),
                                    )

        self.gru_seq = nn.GRU(3330, 128, num_layers=2,dropout=0.2,batch_first=True,bidirectional=True)#两层丢失0.2 128双向
        self.gru_seq1 = nn.GRU(1024, 128, num_layers=1, dropout=0.2, batch_first=True)
        self.gru_seq2 = nn.GRU(1024, 128, num_layers=1, dropout=0.2, batch_first=True)
        self.gru_seq3 = nn.GRU(1024, 128, num_layers=1, dropout=0.2, batch_first=True)
        self.leakyrelu=nn.LeakyReLU()
        self.block1 = nn.Sequential(nn.Linear(4950, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(),
                                    nn.Linear(1024, 256))

        self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool)
        self.dropout = torch.nn.Dropout(self.cnn_dropout)#3c
        self.block1 = nn.Sequential(nn.Linear(12288, 10000),
                                    nn.Linear(10000, 5048),
                                    nn.BatchNorm1d(5048),
                                    nn.LeakyReLU(),
                                    nn.Linear(5048, 2048),
                                    nn.BatchNorm1d(2048),
                                    nn.LeakyReLU(),
                                    nn.Linear(2048,1024),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(),
                                    nn.Linear(1024, 256))
        self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=2,
                                     stride=1
                                     )
        self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1
                                     )
        self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=4,
                                     stride=1
                                     )
        self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=5,
                                     stride=1
                                     )

        self.encoder_layer_word = nn.TransformerEncoderLayer(d_model=150, nhead=3)
        self.transformer_word= nn.TransformerEncoder(self.encoder_layer_word, num_layers=self.num_layer)
        self.embedding_size_word=150
        self.conv1_word = torch.nn.Conv1d(in_channels=self.embedding_size_word,
                                     out_channels=64,
                                     kernel_size=2,
                                     stride=1
                                     )
        self.conv2_word = torch.nn.Conv1d(in_channels=self.embedding_size_word,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1
                                     )
        self.conv3_word = torch.nn.Conv1d(in_channels=self.embedding_size_word,
                                     out_channels=64,
                                     kernel_size=4,
                                     stride=1
                                     )
        self.conv4_word = torch.nn.Conv1d(in_channels=self.embedding_size_word,
                                     out_channels=64,
                                     kernel_size=5,
                                     stride=1
                                     )
        self.MaxPool1d_word = torch.nn.MaxPool1d(kernel_size=self.max_pool)
        self.dropout_word = torch.nn.Dropout(self.cnn_dropout)
        self.gru_word= nn.GRU(384, 25, num_layers=self.gru_yayer, bidirectional=True, dropout=self.gru_dropout)
        self.fc1_word = nn.Sequential(nn.Linear(3200, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(),
                                    nn.Linear(1024, 256),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(),
                                    )
    def TextCNN_word(self, x):
            x1 = self.conv1_word(x)
            x1 = torch.nn.ReLU()(x1)
            x1 = self.MaxPool1d_word(x1)

            x2 = self.conv2_word(x)
            x2 = torch.nn.ReLU()(x2)
            x2 = self.MaxPool1d_word(x2)

            x3 = self.conv3_word(x)
            x3 = torch.nn.ReLU()(x3)
            x3 = self.MaxPool1d_word(x3)

            x4 = self.conv4_word(x)
            x4 = torch.nn.ReLU()(x4)
            x4 = self.MaxPool1d_word(x4)

            y = torch.cat([x1, x2, x3, x4], dim=-1)

            x = self.dropout_word(y)

            # x = x.view(x.size(0), -1)

            return x

    def TextCNN(self, x):
            x1 = self.conv1(x)
            x1 = torch.nn.ReLU()(x1)
            x1 = self.MaxPool1d(x1)

            x2 = self.conv2(x)
            x2 = torch.nn.ReLU()(x2)
            x2 = self.MaxPool1d(x2)

            x3 = self.conv3(x)
            x3 = torch.nn.ReLU()(x3)
            x3 = self.MaxPool1d(x3)

            x4 = self.conv4(x)
            x4 = torch.nn.ReLU()(x4)
            x4 = self.MaxPool1d(x4)

            y = torch.cat([x1, x2, x3, x4], dim=-1)

            x = self.dropout(y)

            # x = x.view(x.size(0), -1)

            return x

    def forward(self, x,pep,word):

        pep = self.embedding(pep)
        pep = self.transformer_encoder_seq(pep).permute(0,2,1)
        pep = self.TextCNN(pep)
        pep, hnn = self.gru(pep)
        pep = pep.reshape(pep.shape[0], -1)
        pep =self.fc1(pep)


        word = self.transformer_word(word).permute(0,2,1)
        word = self.TextCNN_word(word)
        word, hn_word = self.gru_word(word)
        word = word.reshape(word.shape[0], -1)
        word =self.fc1_word(word)

        x=x.reshape(-1, 1, 3330).float()
        x, hn = self.gru_seq(x)
        x=x[:,-1,:]

        x_pep=torch.cat([x, pep,word], dim=1)
        out=self.fc(x_pep)
        out = self.sigmoid(out)
        return out

# 设置超参数
input_size = 1024
hidden_size = 128
num_layers = 2
output_size = 1
epoch = 100
num_heads = 1# 创建模型实例




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

def calculate_metrics(y_true, y_pred_prob):
 #   y_pred = torch.round(y_pred_prob).squeeze().cpu().numpy()  # 将概率转换为二分类输出（0或1）
    y_pred = greater_than_half(y_pred_prob)
    y_true = y_true.cpu().numpy()

    accuracy = metrics.accuracy_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_pred_prob)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)
    specificity = metrics.recall_score(y_true, y_pred, pos_label=0)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

    return accuracy, auc, precision, recall, f1_score, specificity,mcc


def evaluate_model(model, data_loader):
    model.eval()
    y_true_list = []
    y_pred_prob_list = []
    with torch.no_grad():
        for data in data_loader:
            x, pep,word,labels = data
            x = x.to(device)
            pep = pep.to(device)
            word = word.to(device)
            labels = labels.view(-1, 1).float().to(device)

            outputs = model(x,pep,word)
            y_true_list.extend(labels.cpu().numpy())




            y_pred_prob_list.extend(outputs.cpu().numpy())

    y_true_tensor = torch.tensor(y_true_list)
    y_pred_prob_tensor = torch.tensor(y_pred_prob_list)

    accuracy, auc, precision, recall, f1_score, specificity,mcc= calculate_metrics(y_true_tensor,y_pred_prob_tensor)
    return  accuracy, auc, precision, recall, f1_score, specificity,mcc,y_true_tensor,y_pred_prob_tensor



bestparams = None
def trainmodel(cnn_dropout,gru_dropout,gru_layers,trans_head,trans_layers):
        c=cnn_dropout
        e=gru_dropout
        d=gru_layers
        a=trans_head
        b=trans_layers

        accuracies = []
        aurocs = []
        mccs = []
        Spes = []
        SNs = []
        tps = []
        tns = []
        fps = []
        fns = []
        pres = []
        F1s = []
        recs=[]
        kf = KFold(n_splits=10, shuffle=True, random_state=1)

        for t,(train_index, test_index) in enumerate(kf.split(train_x)):
            X1_train, X1_test = train_x[train_index], train_x[test_index]
            X2_train, X2_test = train_pep[train_index], train_pep[test_index]
            X3_train, X3_test =train_word[train_index],train_word[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]

            train_dataset = Data.TensorDataset(X1_train, X2_train,X3_train, y_train)
            test_dataset = Data.TensorDataset(X1_test, X2_test, X3_test,y_test)

            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch, shuffle=False)

            model = LSTMModel(input_size, hidden_size, num_layers, output_size,a,b,c,d,e)
            model.to(device)
            # 定义损失函数和优化器
            criterion = nn.BCELoss().to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.0001)

            best_auc = 0
            best_acc = 0
            rec = 0
            f1 = 0
            spe = 0
            pre = 0
            mcc=0
            acc_auc=0
            n=0
            y_true_tenso1, y_pred_prob_tensor1=0,0

            for i in range(epoch):
                # print("-----第 {} 轮训练开始-----".format(i+1))
                model.train()
                for data in train_dataloader:
                    x,pep,word,labels=data
                    word = word.to(device)
                    pep=pep.to(device)
                    # x = x.reshape(-1, 1, 3330).float()
                    x=x.to(device)

                    labels = labels.view(-1, 1)
                    labels=labels.to(device).float()

                    outputs = model(x,pep,word)
                    loss = criterion(outputs, labels)

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # train_acc, train_auc, train_precision, train_recall, train_f1,train_specificity, train_tp, train_tn, train_fp,train_fn ,train_mcc= evaluate_model(model, train_dataloader)
                test_acc, test_auc, test_precision, test_recall, test_f1, test_specificity,test_mcc,y_true_tenso,y_pred_prob_tensor= evaluate_model(
                    model, val_dataloader)
               #
               #  print(f"Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}, Train Precision: {train_precision:.4f},")
               #  #    f" Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, Train Specificity: {train_specificity:.4f}")
               #  #print(f"Train TP: {train_tp}, Train TN: {train_tn}, Train FP: {train_fp}, Train FN: {train_fn}")
               # # print('\n')
               #  print(f"Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}, Test Precision: {test_precision:.4f},")
                #    f" Tset Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, Test Specificity: {test_specificity:.4f}")
               # print(f"Test TP: {test_tp}, Test TN: {test_tn}, Test FP: {test_fp}, Test FN: {test_fn}")

               # if test_acc > best_acc:
              #  if test_acc+test_mcc+test_auc >acc_auc:
              #   if test_acc + test_auc > best_auc +best_acc:
                if test_auc > best_auc :
                    best_acc = test_acc
                    best_auc = test_auc
                    rec = test_recall
                    f1 = test_f1
                    spe = test_specificity
                    pre = test_precision
                    mcc=test_mcc
                    n=i
                    y_true_tenso1 =y_true_tenso
                    y_pred_prob_tensor1 =  y_pred_prob_tensor


            #        joblib.dump(model, filename='lstmmodel.joblib')
            with open('/home/jwen/neuro2/10zhe/y_pred_prob_tensor{}.pkl'.format(t), 'wb') as f:
                pickle.dump(y_pred_prob_tensor1, f)
            with open('/home/jwen/neuro2/10zhe/y_tru{}.pkl'.format(t), 'wb') as f:
                pickle.dump(y_true_tenso1, f)
            print('\n\n测试结果')
            print('roc:', best_auc)
            print('accuracy:', best_acc)
            print('mcc:',mcc)
            print('pre:', pre)
            print('F1:', f1)
            print('Spe', spe)
            print('rec ',rec)
            print('最佳轮数',n)

            accuracies.append(best_acc)
            aurocs.append(best_auc)
            Spes.append(spe)
            pres.append(pre)
            F1s.append(f1)
            mccs.append(mcc)
            recs.append(rec)
            t+=1
        mean_accuracy = np.mean(accuracies)
        mean_auroc = np.mean(aurocs)
        mean_mcc = np.mean(mccs)
        mean_pre = np.mean(pres)
        mean_F1 = np.mean(F1s)
        mean_Spe = np.mean(Spes)
        mean_rec = np.mean(recs)
        print('\n训练集结果')
        print('mean_accuracy:{:.5f}'.format(mean_accuracy))
        print('mean_auroc:{:.5f}'.format(mean_auroc))
        print('mean_mcc:{:.5f}'.format(mean_mcc))
        print('mean_pre:{:.5f}'.format(mean_pre))
        print('mean_F1:{:.5f}'.format(mean_F1))
        print('mean_Spe:{:.5f}'.format(mean_Spe))
        print('mean_rec :{:.5f}'.format(mean_rec))
        print('模型参数：{} {} {} {} {}'.format(a,b,c,d,e))
        print("----------------------------------")
print('开始')

trainmodel(0.4,0.4,2,2,4)
