import sys
import pickle
from Bio import SeqIO
import numpy as np
from pathlib import Path
import pandas as pd
import warnings
import os
# from tools import read_fasta,supple_X
import gensim
import gzip
import os
import glob
import csv
import multiprocessing
import torch
Path('/home/jwen/word2vec/worddata/train').mkdir(exist_ok=True, parents=True)
Path('/home/jwen/word2vec/worddatatest').mkdir(exist_ok=True, parents=True)
Path('./word2vec_model/').mkdir(exist_ok=True, parents=True)

def generate_pep(file): #读取csv文件并提取序列
    with open(file, 'r') as inf:
        lines = inf.read().splitlines()
    peps = []
    for pep in lines:
        pep, label = pep.split(",")
        peps.append(pep)
    return peps
def calculate_max_sequence_length(file):#计算序列的最大长度
    max_length = 0

    with open(file, 'r') as inf:
        lines = inf.read().splitlines()

    for line in lines:
        pep, _ = line.split(",")  # 假设每行都是蛋白质序列后面跟着逗号和标签
        sequence_length = len(pep)
        max_length = max(max_length, sequence_length)

    return max_length
def tianchong(inputpath,outputpath):
    max_length=calculate_max_sequence_length(inputpath)
    df = pd.read_csv(inputpath, header=None)
    df[0] = df[0].apply(lambda x: x.ljust(max_length, 'X'))
    df.to_csv(outputpath, index=False, header=False)
    a=generate_pep(outputpath)
    return a

input='/home/jwen/NetBCE-main/data/newtotal.csv'
output='output.csv'
# a=tianchong(input,output)
a=generate_pep(output)


#设置参数
word2vec_modell = 'NPs'
Embsize = 150
stride = 1
Embepochs = 50
kmer_len1 = 1
kmer_len2 = 2
kmer_len3 = 3
kmer_len4 = 4
kmer_len5 = 5
kmer_len6 = 6
#定义函数
def Gen_Words(sequences,kmer_len,s):
        out=[]

        for i in sequences:

                kmer_list=[]
                for j in range(0,(len(i)-kmer_len)+1,s):

                            kmer_list.append(i[j:j+kmer_len])

                out.append(kmer_list)

        return out
def read_fasta_file(Filename):
    '''
    used for load fasta data and transformd into numpy.array format
    '''
    fh = open(Filename, 'r')
    seq = []
    for line in fh:
        if line.startswith('>'):
            continue
        else:
            seq.append(line.replace('\n', '').replace('\r', ''))
    fh.close()
    matrix_data = np.array([list(e) for e in seq])
    #print(matrix_data)
    return seq

def train(sequences,kmer_len):
    print('training word2vec modell')
    document= Gen_Words(sequences,kmer_len,stride)
    #print(document)
    modell = gensim.models.Word2Vec (document, window=int(6), min_count=0, vector_size=Embsize,
                                     workers=multiprocessing.cpu_count(),sg=0,sample=33)
    modell.train(document,total_examples=len(document),epochs=Embepochs)
    modell.save('word2vec_model'+'/'+word2vec_modell+str(kmer_len))
    return document
#
#
# #训练word2vec并保存模型
# # all_seq = open('/home/jwen/NetBCE-main/data/newtotal.csv', 'r')
# all_seq = open('/home/jwen/word2vec/output.csv', 'r')
#
# # all_seq = read_fasta_file('/mnt/raid5/data2/zywei/wcross-attention/4mer-data/total.txt')
# # document1 = train(all_seq,kmer_len1)
# document4 = train(all_seq,kmer_len4)
# # document5 = train(all_seq,kmer_len5)
# # document6 = train(all_seq,kmer_len6)
#
# # model1 = gensim.models.Word2Vec.load('word2vec_model'+'/'+word2vec_modell+str(kmer_len1))
# #model2 = gensim.models.Word2Vec.load('word2vec_model'+'/'+word2vec_modell+str(kmer_len2))
# #model3 = gensim.models.Word2Vec.load('word2vec_model'+'/'+word2vec_modell+str(kmer_len3))
# model4 = gensim.models.Word2Vec.load('word2vec_model'+'/'+word2vec_modell+str(kmer_len4))
#
# #读取训练集测试集进行训练词向量
# # x_train1 = pd.read_csv('/home/jwen/NetBCE-main/data/newtotal.csv')
# # x_train2 = x_train1['Sequence'].to_numpy()
# x_train2 =a
# x_train3 = Gen_Words(x_train2,kmer_len4,stride)
# #将训练集通过word2vec--model3进行处理
# X_train = []
# for i in range(0,len(x_train3)):
#     s = []
#     for word in x_train3[i]:
#        s.append(model4.wv[word])
#     X_train.append(s)
# # 将 X_train 转换为 PyTorch 张量
# X_train_tensor = [torch.tensor(sentence) for sentence in X_train]
#
# # 将 X_train_tensor 转换为一个包含所有句子的张量
# X_train_tensor = torch.stack(X_train_tensor)
# with open('/home/jwen/word2vec/词向量数据/all-net-bcell4mer.pkl', 'wb') as f:
#     pickle.dump(X_train_tensor, f)
# print(X_train_tensor.shape)
# print('结束')
def xunlian(input,kmer_len,stride):
    all_seq=open(input, 'r')
    document4 = train(all_seq, kmer_len)
    model=gensim.models.Word2Vec.load('word2vec_model'+'/'+word2vec_modell+str(kmer_len))
    data=generate_pep(output)
    data = Gen_Words(data, kmer_len, stride)
    x_train3=data
    X_train = []
    for i in range(0, len(x_train3)):
        s = []
        for word in x_train3[i]:
            s.append(model.wv[word])
        X_train.append(s)
    X_train_tensor = [torch.tensor(sentence) for sentence in X_train]
    X_train_tensor = torch.stack(X_train_tensor)
    with open('/home/jwen/word2vec/词向量数据/all-net-bcell{}mer.pkl'.format(kmer_len), 'wb') as f:
        pickle.dump(X_train_tensor, f)
    print(X_train_tensor.shape)
    print('结束')
xunlian(output,1,1)
