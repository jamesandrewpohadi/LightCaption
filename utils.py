import pickle
import torch
import torch.nn as nn
from build_vocab import Vocabulary
import time
import numpy as np
from pycocotools.coco import COCO
import random
import os

def get_glove_embedding(vocab,glove_file,dim):
    lines = open('data/glove.6B.{}d.txt'.format(dim)).read().splitlines()
    glove_embedding = {}
    for line in lines:
        token = line.split()
        word = token[0]
        glove_embedding[word] = [float(x) for x in token[1:]]

    e = nn.Embedding(len(vocab.word2idx),dim)
    for w,i in vocab.word2idx.items():
        glove_embed = glove_embedding.get(w,None)
        if glove_embed != None:
            e.weight[i] = torch.tensor(glove_embed)
    torch.save(e.state_dict(),'pretrained/coco2014.glove.6B.{}d.pth'.format(dim))
    print('saved glove embedding in pretrained/coco2014.glove.6B.{}d.pth'.format(dim))

# Generate image ids captions for val test image ids
def generate_val_test_image_ids(val_caption_path,data_folder):
    val_coco = COCO(val_caption_path)
    val_coco_img_ids = list(val_coco.imgs)
    all_idx = random.sample(val_coco_img_ids,2000)
    val_image_ids = all_idx[:1000]
    test_image_ids = all_idx[1000:]
    np.save(os.path.join(data_folder,'test_image_ids.npy'),test_image_ids)
    np.save(os.path.join(data_folder,'val_image_ids.npy'),val_image_ids)
    
def migrate_backbone(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('resnet'):
            k = 'backbone'+k[6:]
        new_state_dict[k] = v
    return new_state_dict
    
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

if __name__ == "__main__":
    vocab = pickle.load(open('vocab.pkl','rb'))
    get_glove_embedding(vocab,'glove.6B.200d.txt',200)