import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
import lightgbm
from Extract_feature import *
import pickle
import sys
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score

def Load_data():
    print('Data Loading...')
    Sequence = []
    with open('fasta.txt', 'r') as myfile:
        for line in myfile:
            if line[0] != '>':
                Sequence.append(line.strip('\n'))

    Mysequence = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        Mysequence.append(zj)
    return Sequence, Mysequence
sequence, sequence1 = Load_data()

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
features_crafted=Get_features(strs,6 )
print('feature_crafted:',len(features_crafted),len(features_crafted[1]))
with open('hand.pkl', 'wb') as f:
    pickle.dump(features_crafted, f)
