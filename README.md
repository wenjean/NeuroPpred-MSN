hand.py are methods for manual feature extraction
t5.py is the method for embedding feature extraction
word2vec2.py is a method for word vectors
dataset.txt is the training and test set
train.py and test.py are the training and testing code, respectively

Because the extracted features cannot be uploaded because the file is too large, you can use the code I uploaded to extract by yourself or contact me directly to obtain, please feel free to contact me. This is my email address 2418538981@qq.com

Follow these steps on how to run NeuroPpred-MSN:
1: Use the PHAT server mentioned in our paper to obtain the secondary structure sequence of the neuropeptide.
2: Configure the conda environment required for NeuroPpred-MSN. Please refer to requirements.txt for the required packages
2: hand.py, t5.py, word2vec.py were used to obtain the hand-crafted features, embedding features, and word vector features of neuropeptides, respectively.
3. Change the paths to the feature files in train.py or test.py to your own.
4. Finally, run train.py or test.py to get the results you need.
