import cPickle as pickle
import tensorflow as tf
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from nltk.corpus import wordnet as wn
import hickle
from core.utils import *
import time 

def main():
    start = time.time()
    with open('./data/word_to_idx.pkl','rb') as f:
        word_to_idx = pickle.load(f)
    with open('./data/top1k.pkl','rb') as f:
        top1k = pickle.load(f)
    captions = {}
    
    top1k = top1k[:5]
    
    for wnid_idx, (wnid,word) in enumerate(top1k):     
        print ' ----- Processing %s, %s, %d / %d' %(wnid, word, wnid_idx, len(top1k)) 
        
        save_path = './data/imagenet/features/%s.hkl' %wnid
        all_feats = hickle.load(save_path)
        data = {}
        data['features'] = all_feats

        model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                                   dim_hidden=1024, n_time_step=16, prev2out=True, 
                                             ctx2out=True, alpha_c=1.0, selector=True, dropout=True)
        solver = CaptioningSolver(model, data, data, n_epochs=15, batch_size=256, update_rule='adam',
                                      learning_rate=0.0025, print_every=2000, save_every=1, image_path='./data/imagenet/features/%s'%wnid,
                                pretrained_model=None, model_path='./data/model/attention', test_model='./data/model/attention/model-18',
                                 print_bleu=False, log_path='./log/')
        
        captions[wnid] = solver.test_imagenet(all_feats)
        tf.get_variable_scope().reuse_variables()
        end = time.time()
        print (end-start)
    
    save_pickle(captions, './data/imagenet_top1k_captions.pkl')
    end = time.time()
    print (end-start)
        
if __name__ == "__main__":
    main()



