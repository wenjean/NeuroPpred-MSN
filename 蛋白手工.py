import torch
from sklearn import metrics
import transformers
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
import lightgbm
import pickle
import sys
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np
from Extract_feature import *

def Load_data():
    with open("/home/jwen/B细胞7871/outputBCELL7871.txt", 'r') as inf:
        lines = inf.read().splitlines()
    peps = []
    for pep in lines:
        peps.append(pep)
        # pep, label = pepp.split(",")
        # peps.append(pep)
    Sequence = peps
    Mysequence = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        Mysequence.append(zj)
    return Sequence, Mysequence
def ALL_features(sequences_Example):
    # Crafted features
    # Automatic extracted features
    tokenizer = T5Tokenizer.from_pretrained("/home/jwen/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("/home/jwen/prot_t5_xl_uniref50")
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
sequence, sequence1 = Load_data()
# featuers_embedding_normalize=ALL_features(sequence1)
# featuers_embedding_normalize =np.array(featuers_embedding_normalize)
# print('特征数量与维度：',featuers_embedding_normalize.shape)
# with open('/home/jwen/B细胞7871/t5.pkl', 'wb') as f:
#         pickle.dump(featuers_embedding_normalize, f)
# sys.exit()
strs = sequence
len_str = len(strs[0])
min_num_index = 0   # 最小值的下标
stack = [strs[0]]   # 利用栈来找出最短的字符串
for index, string in enumerate(strs):
    if len(string) < len_str:
        stack.pop()
        len_str = len(string)
        min_num_index = index # 知道最短字符对应的下标后，也可以自己找出最短字符
        stack.append(string)
print("最短字符串长度:", len_str)
print("最短字符串下标:", min_num_index)
print("最短字符串:", stack)
print("最短字符串:", strs[min_num_index])

unique_letters = set()

for sequence in strs:
    unique_letters.update(sequence)

print(unique_letters)


print(sequence[1],len(sequence[1]))
features_crafted=Get_features(strs,len_str)
print('feature_crafted:',len(features_crafted),len(features_crafted[1]))
with open('/home/jwen/B细胞7871/hand.pkl', 'wb') as f:
    pickle.dump(features_crafted, f)
