from __future__ import absolute_import
from six.moves import cPickle
import gzip
import random
import numpy as np

import glob, os, csv, re
from collections import Counter
import itertools
from keras.preprocessing import sequence

def compute_recall_ks(probas):
    recall_k = {}
    for group_size in [2, 5, 10]:
        recall_k[group_size] = {}
        print ('group_size: %d' % group_size)
        for k in [1, 2, 5]:
            if k < group_size:
                recall_k[group_size][k] = recall(probas, k, group_size)
                print ('recall@%d' % k, recall_k[group_size][k])
    return recall_k

def recall(probas, k, group_size):
    test_size = 10
    n_batches = len(probas) // test_size
    n_correct = 0
    for i in range(n_batches):
        batch = np.array(probas[i*test_size:(i+1)*test_size])[:group_size]
        indices = np.argpartition(batch, -k)[-k:]
        if 0 in indices:
            n_correct += 1
    return float(n_correct) / (len(probas) / test_size)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


#initilize basic vocabulary for cnn, this will change when using features
def init_vocab(emb_size):
    vocabs =['0','S','O','X','-']

    #v2s = list(itertools.product('SOX-', repeat=2))
    #for tupl in v2s:
    #    vocabs.append(''.join(tupl))

    #v3s = list(itertools.product('SOX-', repeat=3))
    #for tupl in v3s:
    #    vocabs.append(''.join(tupl))

    #v4s = list(itertools.product('SOX-', repeat=4))
    #for tupl in v4s:
    #    vocabs.append(''.join(tupl))

    np.random.seed(2017)
    E      = 0.01 * np.random.uniform( -1.0, 1.0, (len(vocabs), emb_size))
    E[0] = 0

    return vocabs, E

#loading the grid with normal CNN
def load_and_numberize_egrids_with_labels(filelist="list_of_grid_pair.txt", maxlen=None, w_size=3, vocabs=None):
    # loading entiry-grid data from list of pos document and list of neg document
    if vocabs is None:
        print("Please input vocab list")
        return None

    list_of_pairs = [line.rstrip('\n') for line in open(filelist)]
    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    labels = []

    
    for pair in list_of_pairs:
        #print(pair)

        pos_doc = pair.split("\t")[0].strip()
        label = pair.split("\t")[1].strip()

        #loading Egrid for POS document
        grid_1 = load_egrid(pos_doc,w_size)
        #grid_0 = load_egrid(neg_doc,w_size)
        

        #if grid_0 != grid_1:
        sentences_1.append(grid_1)
        labels.append(label)
                  
    #assert len(sentences_0) == len(sentences_1)

    vocab_idmap = {}
    for i in range(len(vocabs)):
        vocab_idmap[vocabs[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    #X_0  = numberize_sentences(sentences_0,  vocab_idmap)    

    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=w_size)
    #labels = adjust_index(labels,  maxlen=maxlen, window_size=w_size)

    X_1 = sequence.pad_sequences(X_1, maxlen)
    #labels = sequence.pad_sequences(labels, maxlen)

    return X_1, labels


#loading the grid with normal CNN
def load_and_numberize_egrids(filelist="list_of_grid_pair.txt", maxlen=None, w_size=3, vocabs=None):
    # loading entiry-grid data from list of pos document and list of neg document
    if vocabs is None:
        print("Please input vocab list")
        return None

    list_of_pairs = [line.rstrip('\n') for line in open(filelist)]
    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []

    
    for pair in list_of_pairs:
        #print(pair)

        pos_doc = pair.split("\t")[0]
        neg_doc = pair.split("\t")[1]

        #loading Egrid for POS document
        grid_1 = load_egrid(pos_doc,w_size)
        grid_0 = load_egrid(neg_doc,w_size)
        

        #if grid_0 != grid_1:
        sentences_1.append(grid_1)
        sentences_0.append(grid_0)
                  
    #assert len(sentences_0) == len(sentences_1)

    vocab_idmap = {}
    for i in range(len(vocabs)):
        vocab_idmap[vocabs[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)    

    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=w_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, window_size=w_size)

    X_1 = sequence.pad_sequences(X_1, maxlen)
    X_0 = sequence.pad_sequences(X_0, maxlen)

    return X_1, X_0

#loading the grid with normal CNN from test and dev data
def load_and_numberize_egrids_test_dev(filelist="list_of_grid_pair.txt", maxlen=None, w_size=3, vocabs=None):
    # loading entiry-grid data from list of pos document and list of neg document
    if vocabs is None:
        print("Please input vocab list")
        return None

    list_of_pairs = [line.rstrip('\n') for line in open(filelist)]
    # process postive gird, convert each file to be a sentence
    sentences_1 = []

    
    for pair in list_of_pairs:
        #print(pair)
       
        pos_doc = pair.strip()

        #loading Egrid for POS document
        grid_1 = load_egrid(pos_doc,w_size)
        

        #if grid_0 != grid_1:
        sentences_1.append(grid_1)
                    
        #assert len(sentences_0) == len(sentences_1)

    vocab_idmap = {}
    for i in range(len(vocabs)):
        vocab_idmap[vocabs[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap) 

    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=w_size)

    X_1 = sequence.pad_sequences(X_1, maxlen)

    return X_1

##loading Egrid for a document
def load_egrid(filename,w_size):

    lines = [line.rstrip('\n') for line in open(filename)]
    grid = "0 "* w_size
    for line in lines:
        # merge the grid of positive document 
        e_trans = get_eTrans(sent=line)
        if len(e_trans) !=0:
            grid = grid + e_trans + " " + "0 "* w_size

    return grid


# get each transition for each entity (each line in egrid file)
def get_eTrans(sent=""):
    x = sent.split()
    x = x[1:]
    length = len(x)
    e_occur = x.count('X') + x.count('S') + x.count('O') #counting the number of entities
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""
    return ' '.join(x)


def numberize_sentences(sentences, vocab_idmap):  
    sentences_id=[]  

    for sid, sent in enumerate (sentences):
        tmp_list = []
        #print(sid)
        for wrd in sent.split():
            if wrd in vocab_idmap:
                wrd_id = vocab_idmap[wrd]  
            else:
                wrd_id = 0
            tmp_list.append(wrd_id)

        sentences_id.append(tmp_list)

    return sentences_id  

def adjust_index(X, maxlen=None, window_size=3):
    if maxlen: # exclude tweets that are larger than maxlen
        new_X = []
        for x in X:

            if len(x) > maxlen:
                #print("************* Maxlen of whole dataset: " + str(len(x)) )
                tmp = x[0:maxlen]
                tmp[maxlen-window_size:maxlen] = ['0'] * window_size
                new_X.append(tmp)
            else:
                new_X.append(x)

        X = new_X

    return X



 




