model_name = 'efficientnet.glove.200d'
crop_size=224
vocab_path = 'vocab.pkl'
batch_size = 128
embed_size = 200
hidden_size = 512
learning_rate = 0.001
num_epochs = 5
num_workers = 2
num_layers = 1
log_step = 500
save_step = 3000
# save_step = 200
image_dir = 'resized2014'
model_path = './models'
train_caption_path = '../datasets/coco2014/trainval_coco2014_captions/captions_train2014.json'
test_caption_path = '../datasets/coco2014/trainval_coco2014_captions/captions_val2014.json'

efficientnetBackbone = EfficientNetBackbone.from_pretrained('efficientnet-b0')

encoder = EncoderCNN(efficientnetBackbone,efficientnetBackbone.output_size,embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)

encoder.load_state_dict(torch.