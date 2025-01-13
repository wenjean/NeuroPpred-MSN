Introduction
This repository contains code for "NeuroPpred-MSN: A Neuropeptide Prediction Model Based on Multi-feature Fusion and Siamese Networks".

Paper Abstract
The discovery of neuropeptides offers numerous opportunities for identifying novel drugs and targets to treat a variety of diseases. While various computational methods have been proposed, there remains potential for further performance improvement. In this work, we introduce NeuroPpred-MSN, an innovative and efficient neuropeptide prediction model that leverages multi-feature fusion and Siamese networks. To comprehensively represent the information of the neuropeptides, the peptide sequences were encoded by four encoding schemes (token embedding coding, word2vector coding, embedded features coding, and handcrafted features coding). Then, the token embedding and word2vector embedding were fed to a Siamese network channel. In the other channel of the model, peptide sequences and their secondary structure sequences were fed into the ProtT5-XL-UniRef50 model to generate the embedding features, while handcrafted encoding techniques were used to extract the physicochemical information. Then the two kinds of features are fused and fed into a Bi-GRU network for further processing. Ultimately, the outputs of the two channels are integrated into a fully connected layer, thereby facilitating the generation of the final prediction. The results of the independent test set indicate that NeuroPpred-MSN exhibits superior predictive performance, with an AUROC of 98.3%, exceeding the performance of other state-of-the-art predictors.

Dataset
The dataset in paper is included in data2.csv

Usage

python test.py


