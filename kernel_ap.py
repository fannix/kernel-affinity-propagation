#!/usr/bin/env python2 
from nltk.corpus.reader import TaggedCorpusReader
from nltk.tokenize import RegexpTokenizer
from shogun.Kernel import CommUlongStringKernel
from shogun.Features import StringUlongFeatures, StringCharFeatures, RAWBYTE
from shogun.PreProc import SortUlongString
from scikits.learn.cluster import affinity_propagation
import numpy as np

def read_reviews():
    """
    read reviews from the given file(s).
    """
    from glob import glob
    filenames = glob("input/food*.parsed")
    
    sent_end_pattern = ".\/[,\.]"
    reader = TaggedCorpusReader(
        root = ".",
        fileids = filenames,
        sep = "/",
        sent_tokenizer = RegexpTokenizer(sent_end_pattern, gaps=True))

    li = reader.sents()
    return li

def get_kernel_matrix(li):
    """
    Get kernel matrix from a list of strings.
    """

    order = 3
    gap = 0
    reverse = False
    charfeat = StringCharFeatures(RAWBYTE)
    charfeat.set_features(li)
    feats_train = StringUlongFeatures(charfeat.get_alphabet())
    feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
    preproc = SortUlongString()
    preproc.init(feats_train)
    feats_train.add_preproc(preproc)
    feats_train.apply_preproc()

    use_sign = False

    kernel = CommUlongStringKernel(feats_train, feats_train, use_sign)
    km_train = kernel.get_kernel_matrix()
    return km_train


if __name__ == "__main__":
    li = read_reviews()
    li = [" ".join(e) for e in li if e]
    mat = get_kernel_matrix(li)
    center, labels = affinity_propagation(mat)

    li = np.array(li)
    for i in range(len(center)):
        sents = li[np.where(labels==i)]
        for e in sents:
            print e, "#",
        print i, li[center[i]]
