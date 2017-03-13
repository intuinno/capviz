'''
Great


'''

import pandas as pd
from nltk.corpus import wordnet as wn
from collections import Counter


def _build_top1k_noun():
    tags = pd.read_table('captions.tag', names=['word','pos'])
    nouns = tags[tags.pos == 'NN']
    nouns['lemma'] = nouns.apply(lambda x: wn.morphy(x['word']), axis=1)
    counter = Counter(nouns['lemma'].values)
    #Read imagenet synsets
    with open('./data/imagenet.synsets','r') as f:
        synsets = f.readlines()
    imagenet_synsets = { w.rstrip():True for w in synsets}

    top1k = []
    frequentWords = counter.most_common()
    i = 0
    top1k_dict = {}

    while len(top1k) < 1000:
#        import ipdb; ipdb.set_trace()
        w = frequentWords[i][0]
        if w:
            j = 0
            wnid = 'a'
            while j<len(wn.synsets(w, pos=wn.NOUN)) and wnid not in imagenet_synsets:
                ss = wn.synsets(frequentWords[i][0], pos=wn.NOUN)[j]
                wnid = ss.pos() + str(ss.offset()).zfill(8)
                j += 1
            if wnid in imagenet_synsets and wnid not in top1k_dict:
                top1k.append((wnid,w))
                top1k_dict[wnid] = w
        i += 1
    return top1k


if __name__ == "__main__":
    _build_top1k_noun()

    
