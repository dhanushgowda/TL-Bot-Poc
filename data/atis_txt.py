import gzip
import cPickle
import urllib
import os

def load_udem(filename):
    f = gzip.open(filename,'rb')
    return f

def atisfold(fold):
    assert fold in range(5)
    f = load_udem('atis.fold' + str(fold) + '.pkl.gz')
    train_set, valid_set, test_set, dicts = cPickle.load(f)
    return train_set, valid_set, test_set, dicts


train, _, test, dic = atisfold(3)

w2idx, ne2idx, labels2idx = dic['words2idx'], dic['tables2idx'], dic['labels2idx']

idx2w = dict((v, k) for k, v in w2idx.iteritems())
idx2ne = dict((v, k) for k, v in ne2idx.iteritems())
idx2la = dict((v, k) for k, v in labels2idx.iteritems())

sentence_indexes = test[0] + train[0] + _[0]
# f = open('sentences.txt','w')
#
#
for i in sentence_indexes[0:1]:
#     # print i
    sen = ' '.join(map(lambda x: idx2w[x], i))
#     f.write(sen + '\n')  # python will convert \n to os.linesep
    print sen
#
# f.close()
