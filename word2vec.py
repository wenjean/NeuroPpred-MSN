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
Path('./worddata/train').mkdir(exist_ok=True, parents=True)
Path('./worddatatest').mkdir(exist_ok=True, parents=True)
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
        # pep = line #没有标签
        # sequence_length = len(pep)
        sequence_length = len(pep)
        max_length = max(max_length, sequence_length)

    return max_length
def tianchong(inputpath,outputpath):
    max_length=calculate_max_sequence_length(inputpath)
    print('最大序列长度：{}'.format(max_length))
    df = pd.read_csv(inputpath, header=None)
    df[0] = df[0].apply(lambda x: x.ljust(max_length, 'X'))
    df.to_csv(outputpath, index=False, header=False)
    a=generate_pep(outputpath)
    return a
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
def xunlian(input,kmer_len,stride):
    all_seq=open(input, 'r',encoding='latin-1')
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
    with open('{}mer.pkl'.format(kmer_len), 'wb') as f:
        pickle.dump(X_train_tensor, f)
    print(X_train_tensor.shape)
    print('结束')

input='data2.csv'
output='data3.csv'
a=tianchong(input,output)


word2vec_modell = 'NPs'
Embsize = 150
stride = 1
Embepochs = 50

xunlian('data3.csv',1,1)
xunlian('data3.csv',4,1)

