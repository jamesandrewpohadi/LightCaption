# Test
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

import nltk
from nltk.translate.bleu_score import corpus_bleu
from utils import Timer
from tqdm import tqdm
import pickle
from PIL import Image
from pycocotools.coco import COCO
from build_vocab import Vocabulary
from models import EfficientNetBackbone, Encoder, Decoder
from utils import migrate_backbone
import numpy as np

import argparse

def test(args):
    with open(args.vocab, 'rb') as f:
        vocab = pickle.load(f)
        
    encoderTimer = Timer()
    decoderTimer = Timer()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnet152 = models.resnet152(pretrained=True)
    resnet_output_dim = resnet152.fc.in_features
    modules = list(resnet152.children())[:-1]      # delete the last fc layer.
    resnet152 = nn.Sequential(*modules)

    encoder = Encoder(resnet152,resnet_output_dim,args.embed).to(device)
    decoder = Decoder(args.embed,args.hidden, len(vocab), args.layer).to(device)
    encoder.load_state_dict(migrate_backbone(torch.load(args.encoder)))
    decoder.load_state_dict(torch.load(args.decoder))
    encoder.eval()
    decoder.eval()

    transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                (0.229, 0.224, 0.225))])

    coco = COCO(args.caption)

    total_bleu_score = 0
    count = 0
    references = []
    sentences = []

    test_image_ids = np.load('data/val_image_ids.npy')
    pbar = tqdm(test_image_ids)
    for image_id in pbar:
        image_id_string = str(image_id)
        image_id_string = '0'*(6-len(image_id_string))+image_id_string
        image = Image.open('../datasets/coco2014/val2014/COCO_val2014_000000{}.jpg'.format(image_id_string)).convert('RGB') 
        image = image.resize([args.resize, args.resize], Image.LANCZOS)
        image = transform(image).unsqueeze(0)
        image_tensor = image.to(device)
        encoderTimer.tic()
        feature = encoder(image_tensor)
        encoderTimer.toc()
        decoderTimer.tic()
        sampled_ids = decoder.sample(feature)
        decoderTimer.toc()
        sampled_ids = sampled_ids.squeeze(0).cpu().numpy()          # (1, max_seq_length) -> (max_seq_length) 
        # Convert word_ids to words
        caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            if word == '<start>':
                continue
            if word == '<end>':
                break
            caption.append(word)
            
        pbar.set_description('Inference Time, Encoder: {:.3f}s, Decoder: {:.3f}s'.format(encoderTimer.average_time,decoderTimer.average_time))
        
        reference = [nltk.tokenize.word_tokenize(cap['caption'].lower()) for cap in coco.imgToAnns[image_id]]
        sentences.append(caption)
        references.append(reference)
        
    print('Average Inference Time, Decoder: {:.3f}s, Encoder: {:.3f}s'.format(encoderTimer.average_time,decoderTimer.average_time))

    if torch.cuda.is_available():
        print('GPU: {}'.format(torch.cuda.get_device_name(0)))
    bleu1 = corpus_bleu(references, sentences, weights=(1,0,0,0))
    bleu2 = corpus_bleu(references, sentences, weights=(0.5,0.5,0,0))
    bleu3 = corpus_bleu(references, sentences, weights=(0.333,0.333,0.333,0))
    bleu4 = corpus_bleu(references, sentences, weights=(0.25,0.25,0.25,0.25))
    print("BLEU-1: {:.3f}".format(bleu1))
    print("BLEU-2: {:.3f}".format(bleu2))
    print("BLEU-3: {:.3f}".format(bleu3))
    print("BLEU-4: {:.3f}".format(bleu4))

if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser(description='test_glove.200d.py')
    parser.add_argument('-l','--layer',default=1,type=int)
    parser.add_argument('-hd','--hidden',default=512,type=int)
    parser.add_argument('-embed','--embed',default=200,type=int)
    parser.add_argument('-v','--vocab',default='data/vocab.pkl')
    parser.add_argument('-c','--caption',default='../datasets/coco2014/trainval_coco2014_captions/captions_val2014.json')
    # parser.add_argument('-e','--encoder',default='models/encoder.glove.200d-5-3000.pth')
    # parser.add_argument('-d','--decoder',default='models/decoder.glove.200d-5-3000.pth')
    parser.add_argument('-e','--encoder',default='models/encoder.glove.200d.std1-5-3000.pth')
    parser.add_argument('-d','--decoder',default='models/decoder.glove.200d.std1-5-3000.pth')
    # parser.add_argument('-e','--encoder',default='models/no-retrain-embedding-encoder-5-3000.ckpt')
    # parser.add_argument('-d','--decoder',default='models/no-retrain-embedding-decoder-5-3000.ckpt')
    parser.add_argument('-r','--resize',default=224,type=int)
    args = parser.parse_args()
    test(args)