import json
import os
import pandas as pd
import numpy as np
import cPickle as pickle
import hickle
from collections import Counter
from nltk.corpus import stopwords 
from nltk.corpus import wordnet as wn
import urllib
import tarfile
from PIL import Image
from core.vggnet import Vgg19
import tensorflow as tf
from scipy import ndimage
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import *
from datetime import datetime

def _process_caption_data(caption_file, image_dir, max_length):
    with open(caption_file) as f:
        caption_data = json.load(f)

    # id_to_filename is a dictionary such as {image_id: filename]} 
    id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}

    # data is a list of dictionary which contains 'captions', 'file_name' and 'image_id' as key.
    data = []
    for annotation in caption_data['annotations']:
        image_id = annotation['image_id']
        annotation['file_name'] = os.path.join(image_dir, id_to_filename[image_id])
        data += [annotation]

    # convert to pandas dataframe (for later visualization or debugging)
    caption_data = pd.DataFrame.from_dict(data)
    del caption_data['id']
    caption_data.sort_values(by='image_id', inplace=True)
    caption_data = caption_data.reset_index(drop=True)

    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        caption = caption.replace('.', '').replace(',', '').replace("'", "").replace('"', '')
        caption = caption.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ')
        caption = " ".join(caption.split())  # replace multiple spaces

        caption_data.set_value(i, 'caption', caption.lower())
        if len(caption.split(" ")) > max_length:
            del_idx.append(i)

    # delete captions if size is larger than max_length
    print "The number of captions before deletion: %d" % len(caption_data)
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print "The number of captions after deletion: %d" % len(caption_data)
    return caption_data

def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ') # caption contrains only lower-case words
        for w in words:
            counter[w] +=1
        
        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print "Max length of caption: ", max_len
    return word_to_idx

def _build_top1k_vocab(annotations):
    counter = Counter()
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ') # caption contrains only lower-case words
        for w in words:
            counter[w] +=1
    #Read imagenet synsets
    with open('./data/imagenet.synsets','r') as f:
        synsets = f.readlines()
    imagenet_synsets = { w.rstrip():True for w in synsets}
    
    top1k = []
    frequentWords = counter.most_common()
    i = 0
    top1k_dict = {}
    
    while len(top1k) < 1000:
        w = frequentWords[i][0]
        ss = wn.synsets(frequentWords[i][0])
        j = 0
        while j< len(ss) and ss[j].pos() != 'n': j += 1
        if j<len(ss):
            wnid = ss[j].pos() + str(ss[j].offset()).zfill(8)
            if wnid in imagenet_synsets and wnid not in top1k_dict:
                top1k.append((wnid,w))
                top1k_dict[wnid] = w
        i += 1
    return top1k

def _build_top1k_noun():
    tags = pd.read_table('captions.tag', names=['word','pos'])
    nouns = tags[tags.pos == 'NN']
    nouns['lemma'] = nouns.apply(lambda x: wn.morphy(x['word']), axis=1)
    c = Counter(nouns['lemma'].values)
    #Read imagenet synsets
    with open('./data/imagenet.synsets','r') as f:
        synsets = f.readlines()
    imagenet_synsets = { w.rstrip():True for w in synsets}
    
    top1k = []
    frequentWords = counter.most_common()
    i = 0
    top1k_dict = {}
    
    while len(top1k) < 1000:
        w = frequentWords[i][0]
        if w:
            j = 0
            wnid = 'a'
            while j<len(wn.synsets(w, pos=wn.NOUN)) and wnid not in imagenet_synsets:
                ss = wn.synset(frequentWords[i][0]+'.n.'+str(j+1).zfill(2))
                wnid = ss.pos() + str(ss.offset()).zfill(8)
                j += 1
            if wnid not in top1k_dict:
                top1k.append((wnid,w))
                top1k_dict[wnid] = w
        i += 1
    return top1k

def resize_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) /2
        bottom = height - top
        left = 0
        right = width 
    image = image.crop((left, top, right, bottom))
    image = image.resize([224,224], Image.ANTIALIAS)
    return image

def main():
    
    start = datetime.now()
    
    caption_file = 'data/annotations/captions_train2014.json'
    image_dir = 'image/train2014_resized'
    max_length = 15
    word_count_threshold = 1
    vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'
    batch_size = 50
    
    print '1. Building Top 1K dictionary from Train dataset'
    
    if not os.path.exists('./data/top1k.pkl'):
        train_dataset = _process_caption_data(caption_file=caption_file,
                                          image_dir=image_dir,
                                          max_length=max_length)
        word_to_idx = _build_vocab(annotations=train_dataset, threshold=word_count_threshold)
        save_pickle(word_to_idx, './data/word_to_idx.pkl')
        top1k = _build_top1k_noun()
        save_pickle(top1k, './data/top1k.pkl')
    else:
        top1k = load_pickle('./data/top1k.pkl')
    
    print '2. Download and Process each keywords'
    
    cur_dir = os.getcwd()    
    wnid_idx = 0
    vggnet = Vgg19(vgg_model_path)
    vggnet.build()
    captions = {}
    
    if not os.path.exists('./data/imagenet/features/'):
        os.makedirs('./data/imagenet/features/')
    for wnid_idx, (wnid,word) in enumerate(top1k):   
        save_path = './data/imagenet/features/%s.list' %wnid
        if not os.path.exists(save_path):
            print ' ----- Processing %s, %s, %d / %d' %(wnid, word, wnid_idx, len(top1k)) 
            print '\tdownloading'
            pre_url = 'http://www.image-net.org/download/synset?wnid='
            post_url = '&username=intuinno&accesskey=6be8155ee3d56b5120241b3bda13412d3cc0cd42&release=latest&src=stanford'
            testfile = urllib.URLopener()
            try:
                testfile.retrieve(pre_url+wnid+post_url, wnid+'.tar')
            except IOError as e:
                print 'Failed to download'
            else:

                original_dir = './data/imagenet/photos/%s/original/'%wnid
                resized_dir = './data/imagenet/photos/%s/resized/'%wnid

                if not os.path.exists(original_dir):
                    os.makedirs(original_dir)
                    os.makedirs(resized_dir)
                    os.rename(wnid+'.tar', original_dir + 'data.tar' )
                    os.chdir(original_dir)
                    tar = tarfile.open('data.tar')
                    tar.extractall()
                    tar.close()
                    os.remove('data.tar')
                    os.chdir(cur_dir)
                else:
                    os.remove('%s.tar'%wnid)

                print '\tresizing'
                resized_files = []
                image_files = os.listdir(original_dir)
                for i, image_file in enumerate(image_files):
                #     from IPython.core.debugger import Tracer; Tracer()() 
                    try:
                        image = Image.open(os.path.join(original_dir, image_file))
                    except IOError as e:
                        print 'Error: cannot open %s' %(os.path.join(original_dir, image_file))
                    else:               
                        image = resize_image(image)
                        image.save(os.path.join(resized_dir, image_file), image.format)
                        resized_files.append(image_file)

                image_files = resized_files
                print '\tget vgg19 image features'
                with tf.Session() as sess:
                    tf.initialize_all_variables().run()
                    n_examples = len(image_files)
                    all_feats = np.ndarray([n_examples, 196,512], dtype=np.float32)

                    for start, end in zip(range(0, n_examples, batch_size),
                                          range(batch_size, n_examples+batch_size, batch_size)):
                        image_batch_file = image_files[start:end]
                        image_batch = np.array(map(lambda x: ndimage.imread(os.path.join(resized_dir, x), mode='RGB'), image_batch_file)).astype(np.float32)
                        feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                        all_feats[start:end, :] = feats

                save_path = './data/imagenet/features/%s.hkl' %wnid
                hickle.dump(all_feats, save_path)
                save_path = './data/imagenet/features/%s.list' %wnid
                save_pickle(image_files, save_path)

                print "\tSaved %s.." % save_path

    
if __name__ == "__main__":
    main()



