import numpy
import time
import sys
import subprocess
import os
import random

from data import load
from rnn.elman import model
from metrics.accuracy import conlleval
from utils.tools import shuffle, minibatch, contextwin


if __name__ == '__main__':

    s = {'fold':3, # 5 folds 0,1,2,3,4
         'lr':0.0627142536696559,
         'verbose':1,
         'decay':False, # decay on the learning rate if improvement stops
         'win':7, # number of words in the context window
         'bs':9, # number of backprop through time steps
         'nhidden':100, # number of hidden units
         'seed':345,
         'emb_dimension':100, # dimension of word embedding
         'nepochs':50}

    folder = "elman-forward"
    # if not os.path.exists(folder): os.mkdir(folder)

    # load the dataset
    train_set, valid_set, test_set, dic = load.atisfold(s['fold'])
    idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dic['words2idx'].iteritems())
    word2idx = dic['words2idx']
    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex,  test_ne,  test_y  = test_set

    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_lex)

    # instanciate the model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])
    rnn = model(    nh = s['nhidden'],
                    nc = nclasses,
                    ne = vocsize,
                    de = s['emb_dimension'],
                    cs = s['win'] )

    rnn.emb.set_value(numpy.load('elman-forward/embeddings.npy'))
    rnn.b.set_value(numpy.load('elman-forward/b.npy'))
    rnn.bh.set_value(numpy.load('elman-forward/bh.npy'))
    rnn.h0.set_value(numpy.load('elman-forward/h0.npy'))
    rnn.W.set_value(numpy.load('elman-forward/W.npy'))
    rnn.Wh.set_value(numpy.load('elman-forward/Wh.npy'))
    rnn.Wx.set_value(numpy.load('elman-forward/Wx.npy'))

    sample = "please find 9 flights from los angeles to miami arriving at 9".split()
    #temp = [ map(lambda x: word2idx[x], w) for w in sample]
    temp=[]
    for tem in sample:
        if tem in word2idx:
            temp.append(word2idx[tem])
        else:
            temp.append(word2idx['<UNK>'])
    temp_nd = numpy.ndarray((1,len(temp)),buffer=numpy.array(temp),dtype=int)
    # evaluation // back into the real world : idx -> words
    predictions_test = [ map(lambda x: idx2label[x], rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32'))) for x in temp_nd ]
    print predictions_test

    predictions_test = [map(lambda x: idx2label[x], \
                            rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32'))) \
                        for x in test_lex]
    groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
    words_test = [map(lambda x: idx2word[x], w) for w in test_lex]

    predictions_valid = [map(lambda x: idx2label[x], \
                             rnn.classify(numpy.asarray(contextwin(x, s['win'])).astype('int32'))) \
                         for x in valid_lex]
    groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
    words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]

    # evaluation // compute the accuracy using conlleval.pl
    res_test = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
    res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')

    print res_test, res_valid