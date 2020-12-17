import os
import torch
from torch import nn
from model import EncoderCNN,DecoderRNN
import numpy as np
from get_loader import get_loader
import torchvision.transforms as transforms
from PIL import Image
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
emb_size = 512
hid_dim = 512
transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ])
dataset,loader =  get_loader('Flicker8k_Dataset','captions.txt',transforms=transforms)
vocab_size = len(dataset.vocab)


encoder_file = 'encoder-19.pkl' # trained for 20 epochs
decoder_file = 'decoder-19.pkl' # trained for 20 epochs


encoder = EncoderCNN(emb_size)
encoder.eval()
decoder = DecoderRNN(emb_size,hid_dim,vocab_size)
decoder.eval()

# Load last saved weights of encoder and decoder and then pass encoder and decoder to device
encoder.load_state_dict(torch.load(os.path.join('/models',encoder_file)))
decoder.load_state_dict(torch.load(os.path.join('/models',decoder_file)))

# Pass encoder and decoder to device(if CUDA is available)
encoder.to(device)
decoder.to(device)


img_file = '/content/drive/MyDrive/Image_captioning/test_images/horse.png'
img_transform = transform(Image.open(img_file).convert('RGB')).unsqueeze(0)

features = encoder(img_transform.to(device)).unsqueeze(1) # add extra dimension to features
print(decoder.sample(features,dataset.vocab))














