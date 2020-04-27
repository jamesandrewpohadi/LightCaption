import json

from flask import Flask, request, render_template
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import pickle
from PIL import Image
from models import EfficientNetBackbone, Encoder, DecoderScaleDown
import numpy as np

app = Flask(__name__)

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

# Web service initialization
@app.route('/')
def index():
    return render_template('index.html',label=False)

# @app.route('/')
@app.route('/', methods=['POST'])
def send_image():
    file = request.files['image']
    img_src = request.values['img_src']
    if not file:
        return render_template('index.html', label=False)
    image = Image.open(file).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    image = transform(image).unsqueeze(0)
    image_tensor = image.to(device)
    feature = encoder(image_tensor)
    cell_state = feature.unsqueeze(1)
    hidden_state = feature.unsqueeze(1)
    start = torch.tensor(vocab('<start>')).to(device)
    inputs = decoder.embed(start).unsqueeze(0)
    sampled_ids = decoder.sample(inputs,(hidden_state,cell_state),'temperature')
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
    caption = ' '.join(caption)
    return render_template('index.html',caption=caption,img_src=img_src,label=True)

if __name__ == '__main__':
    with open('data/vocab10.pkl', 'rb') as f:
        vocab = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    efficientnetBackbone = EfficientNetBackbone.from_pretrained('efficientnet-b1')
    encoder = Encoder(efficientnetBackbone,efficientnetBackbone.output_size,256).to(device)
    decoder = DecoderScaleDown(200,256, len(vocab), 1).to(device)
    encoder.load_state_dict(torch.load('models/encoder.efficientnetb1-hidden256-connected_cell_hidden-vocab10-scale_down3-6-2000.pth'))
    decoder.load_state_dict(torch.load('models/decoder.efficientnetb1-hidden256-connected_cell_hidden-vocab10-scale_down3-6-2000.pth'))
    encoder.eval()
    decoder.eval()

    transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                (0.229, 0.224, 0.225))])
    app.run(host='0.0.0.0', port=8080)