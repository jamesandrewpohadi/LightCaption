# libraries
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
import torch.utils.data as data

import os
import pickle
import numpy as np
import nltk
from PIL import Image
from pycocotools.coco import COCO
from build_vocab import Vocabulary
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from models import EfficientNetBackbone, Encoder, Decoder, DecoderAttent
from nltk.translate.bleu_score import corpus_bleu
import shutil
from tqdm import tqdm
import random
from utils import Timer
import argparse

from telegram import Bot

bot = Bot('1045358491:AAF42Ermr_8YCjtmRNdG4m3SOU3j2Mggigk')
    
# class Attention
    
class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None,start=True):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        self.start = start

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        if self.start:
            caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, start, shuffle, num_workers, collate_fn):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform,
                       start=start)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader

def visualise_one(coco, encoder, decoder, vocab, device):
    print('visualise one')
    encoder.eval()
    decoder.eval()
    transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                (0.229, 0.224, 0.225))])
    rand_idx = random.randint(0,len(coco.imgs)-1)
    image_id = list(coco.imgs.keys())[rand_idx]
    string_id = '0'*(6-len(str(image_id)))+str(image_id)
    image = Image.open('../datasets/coco2014/val2014/COCO_val2014_000000{}.jpg'.format(string_id)).convert('RGB')
    
    image = image.resize([args.target_size, args.target_size], Image.LANCZOS)
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    feature = encoder(image_tensor)
    cell_state = feature.unsqueeze(0)
    hidden_state = torch.zeros(cell_state.shape).to(device)
    start = torch.tensor(vocab('<start>')).to(device)
    inputs = decoder.embed(start).unsqueeze(0)
    sampled_ids = decoder.sample(inputs,(hidden_state,cell_state))
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<start>':
                    continue
        if word == '<end>':
            break
        sampled_caption.append(word)
    sentence = ' '.join(sampled_caption[:-1])
    plt.imshow(image)
    plt.xlabel(sentence)
    plt.savefig('sample.png')
    plt.clf()
    encoder.train()
    decoder.train()

def val(encoder,decoder,device,coco,vocab,args):
    encoder.eval()
    decoder.eval()
        
    encoderTimer = Timer()
    decoderTimer = Timer()

    transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                (0.229, 0.224, 0.225))])
    references = []
    sentences = []
    test_image_ids = np.load('data/test_image_ids.npy')
    pbar = tqdm(test_image_ids)
    with torch.no_grad():
        for image_id in pbar:
            image_id_string = str(image_id)
            image_id_string = '0'*(6-len(image_id_string))+image_id_string
            image = Image.open('../datasets/coco2014/val2014/COCO_val2014_000000{}.jpg'.format(image_id_string)).convert('RGB') 
            image = image.resize([args.target_size, args.target_size], Image.LANCZOS)
            image = transform(image).unsqueeze(0)
            image_tensor = image.to(device)
            encoderTimer.tic()
            feature = encoder(image_tensor)
            encoderTimer.toc()
            cell_state = feature.unsqueeze(1)
            hidden_state = feature.unsqueeze(1)
            # hidden_state = torch.zeros(cell_state.shape).to(device)
            decoderTimer.tic()
            start = torch.tensor(vocab('<start>')).to(device)
            inputs = decoder.embed(start).unsqueeze(0)
            sampled_ids = decoder.sample(inputs,(hidden_state,cell_state))
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
    bleu1 = corpus_bleu(references, sentences, weights=(1,0,0,0))
    bleu2 = corpus_bleu(references, sentences, weights=(0.5,0.5,0,0))
    bleu3 = corpus_bleu(references, sentences, weights=(0.333,0.333,0.333,0))
    bleu4 = corpus_bleu(references, sentences, weights=(0.25,0.25,0.25,0.25))
    print("BLEU-1: {:.3f} | BLEU-2: {:.3f} | BLEU-3: {:.3f} | BLEU-4: {:.3f}".format(bleu1,bleu2,bleu3,bleu4))
    return bleu1,bleu2,bleu3,bleu4

def main(args):

    bot.sendMessage(-428968689,'Training {}'.format(args.name))

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Load vocabulary wrapper
    with open(args.vocab, 'rb') as f:
        vocab = pickle.load(f)

    val_coco = COCO(args.val_caption)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([ 
        transforms.RandomCrop(args.target_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                            (0.229, 0.224, 0.225))])

    # Build data loader
    train_data_loader = get_loader(args.image, args.train_caption, vocab, 
                        train_transform, args.batch_size,start=False,
                        shuffle=True, num_workers=args.num_workers,
                        collate_fn=collate_fn)

    # Model architechture
    efficientnetBackbone = EfficientNetBackbone.from_pretrained('efficientnet-b0')

    encoder = Encoder(efficientnetBackbone,efficientnetBackbone.output_size,args.hidden).to(device)
    decoder = Decoder(args.embed, args.hidden, len(vocab), args.layer).to(device)
    # decoder.embed.load_state_dict(torch.load('pretrained/coco2014.glove.6B.200d.std1.pth'))

    # encoder.load_state_dict(torch.load('models/encoder.efficientnetb0-glove.200d.std1-hidden512-connected_cell-attent-1-12000.pth'))
    # decoder.load_state_dict(torch.load('models/decoder.efficientnetb0-glove.200d.std1-hidden512-connected_cell-attent-1-12000.pth'))

    encoder.train()
    decoder.train()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Training
    total_step = len(train_data_loader)
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch+1,args.num_epochs))
        for i, (images, captions, lengths) in tqdm(enumerate(train_data_loader)):

            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            features = encoder(images)
            cell_state = features.unsqueeze(0)
            hidden_state = features.unsqueeze(0)
            # hidden_state = torch.zeros(cell_state.shape).to(device)
            batch_size = features.shape[0]
            start = torch.tensor(vocab('<start>')).to(device)
            inputs = decoder.embed(start).repeat(batch_size,1)
            outputs = decoder(inputs, captions, lengths, (hidden_state,cell_state))
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                    .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                bot.sendMessage(-428968689,'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                    .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 

            if (i+1) % args.visualise_step == 0:
                visualise_one(val_coco, encoder, decoder, vocab, device)
                img = open('sample.png','rb')
                bot.sendPhoto(-428968689,img)
            
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder.{}-{}-{}.pth'.format(args.name, epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder.{}-{}-{}.pth'.format(args.name, epoch+1, i+1)))
        val(encoder,decoder,device,val_coco,vocab,args)
        encoder.train()
        decoder.train()
    

if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser(description='test_glove.200d.std1')

    # model config
    parser.add_argument('-l','--layer',default=1,type=int)
    parser.add_argument('-hd','--hidden',default=512,type=int)
    parser.add_argument('-embed','--embed',default=200,type=int)

    # training config
    parser.add_argument('-e','--num_epochs',default=15,type=int)
    parser.add_argument('-b','--batch_size',default=32,type=int)
    parser.add_argument('-lr','--learning_rate',default=0.001,type=int)
    parser.add_argument('-nw','--num_workers',default=2,type=int)
    parser.add_argument('-ss','--save_step',default=12000,type=int)
    parser.add_argument('-vs','--visualise_step',default=400,type=int)
    parser.add_argument('-ls','--log_step',default=500,type=int)

    # data folders
    parser.add_argument('-v','--vocab',default='data/vocab10.pkl')
    parser.add_argument('-i','--image',default='data/resized2014')
    parser.add_argument('-tc','--train_caption',default='../datasets/coco2014/trainval_coco2014_captions/captions_train2014.json')
    parser.add_argument('-vc','--val_caption',default='../datasets/coco2014/trainval_coco2014_captions/captions_val2014.json')
    parser.add_argument('-name','--name',default="efficientnetb0-glove.200d.std1-hidden512-connected_cell_hidden-vocab10")
    parser.add_argument('-mp','--model_path',default="./models")
    parser.add_argument('-t','--target_size',default=224,type=int)
    args = parser.parse_args()

    shutil.copy('train.py','methods/train_{}.py'.format(args.name))
    main(args)          