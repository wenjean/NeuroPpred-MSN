import torch
from sklearn import metrics
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
import pickle
import sys
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np

def Load_data():
    import pandas as pd
    # 读取CSV文件，假设第一列是序列，第二列是标签
    data= pd.read_csv('train.csv', header=None)
    import numpy as np
    # （1）获取第1列
    Sequence = data.iloc[1:,0]
    Sequence =np.array(Sequence)
    Mysequence = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i])-1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        Mysequence.append(zj)
    label = data.iloc[1:,1]
    label = np.array(label)
    return Mysequence, label

def ALL_features(sequences_Example):
    # Crafted features
    # Automatic extracted features
    tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50")
    gc.collect()
    print(torch.cuda.is_available())
    # 'cuda:0' if torch.cuda.is_available() else
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    for i in range(len(sequences_Example)):
        print('For sequence ', str(i+1))
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    print(len(features_normalize), len(features_normalize[0]))
    return features_normalize
Mysequence, label_embedding=Load_data()

with open('t5.pkl', 'rb') as f:
    loaded_sequences = pickle.load(f)
featuers_embedding_normalize=ALL_features(loaded_sequences)
featuers_embedding_normalize =np.array(featuers_embedding_normalize)
print('特征数量与维度：',featuers_embedding_normalize.shape)
with open('bce-train.pkl', 'wb') as f:
         pickle.dump(featuers_embedding_normalize, f)
#print('标签数量：',label_embedding.shape)
# with open('test-featuers_embedding_normalize.pkl', 'wb') as f:
#         pickle.dump(featuers_embedding_normalize, f)
# with open('test-label_embedding.pkl', 'wb') as f:
#         pickle.dump(label_embedding, f)
