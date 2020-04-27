import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from efficientnet_pytorch import EfficientNet
from torch.distributions.categorical import Categorical

class EfficientNetBackbone(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None):
        EfficientNet.__init__(self, blocks_args, global_params)
        self.output_size = self._fc.in_features
        self._fc = None
        
    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        return x

class Encoder(nn.Module):
    def __init__(self, backbone, backbone_output_dim, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Encoder, self).__init__()
        self.backbone = backbone
        self.linear = nn.Linear(backbone_output_dim, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.backbone(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, inputs, captions, lengths,states=None):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((inputs.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        
        hiddens, _ = self.lstm(packed,states)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample_greedy(self,inputs,states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = inputs.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

    def sample_temperature(self,inputs,states=None):
        sampled_ids = []
        inputs = inputs.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            predicted = Categorical(outputs).sample()
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
    
    def sample_beamSearch(self,features,states=None):
        pass
    
    def sample(self, inputs, states=None,method='greedy'):
        if method=='greedy':
            return self.sample_greedy(inputs,states)
        elif method=='temperature':
            return self.sample_temperature(inputs,states)

class DecoderScaleDown(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20,scale_down_size=3):
        """Set the hyper-parameters and build the layers."""
        super(DecoderScaleDown, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.scale_down = nn.Linear(hidden_size, hidden_size//scale_down_size)
        self.linear = nn.Linear(hidden_size//scale_down_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths,states=None):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        
        hiddens, _ = self.lstm(packed,states)
        outputs = self.linear(self.scale_down(hiddens[0]))
        return outputs

    def sample_greedy(self,features,states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(self.scale_down(hiddens.squeeze(1)))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

    def sample_temperature(self,features,states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        softmax = nn.Softmax(dim=1)
        temperature = 0.5
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(self.scale_down(hiddens.squeeze(1)))            # outputs:  (batch_size, vocab_size)
            predicted = Categorical(softmax(outputs/temperature)).sample()
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
    
    def sample(self, features, states=None,method='greedy'):
        if method=='greedy':
            return self.sample_greedy(features,states)
        elif method=='temperature':
            return self.sample_temperature(features,states)