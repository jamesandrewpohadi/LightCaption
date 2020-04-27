
crop_size=224
vocab_path = 'vocab.pkl'
batch_size = 128
embed_size = 256
hidden_size = 512
learning_rate = 0.001
num_epochs = 5
num_workers = 2
num_layers = 1
log_step = 100
save_step = 3000
image_dir = 'resized2014'
model_path = './models'
train_caption_path = '../datasets/coco2014/trainval_coco2014_captions/captions_train2014.json'
test_caption_path = '../datasets/coco2014/trainval_coco2014_captions/captions_val2014.json'

resnet152 = models.resnet152(pretrained=True)
resnet_output_dim = resnet152.fc.in_features
modules = list(resnet152.children())[:-1]      # delete the last fc layer.
resnet152 = nn.Sequential(*modules)

encoder = EncoderCNN(resnet152,resnet_output_dim,embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)

encoder.load_state_dict(torch.load('models/encoder.glove.200d-5-3000.pth'))
decoder.load_state_dict(torch.load('models/decoder.glove.200d-5-3000.pth'))