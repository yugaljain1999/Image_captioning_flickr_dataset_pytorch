import shutil
import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from utils import save_checkpoint, load_checkpoint
from get_loader import get_loader
from model import CNNtoRNN

#from tqdm import tqdm
def train(only_vocab = False):
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ])

    dataset,train_loader = get_loader(
        root_dir="Flicker8k_Dataset",
        caption_file="captions.txt",
        transforms=transform
    )   

    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_model = True
    save_model = False
    train_CNN = False


    embed_size = 512
    hidden_size = 512
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 5
    

    writer = SummaryWriter("runs/flickr")
    step = 0

    if only_vocab:
      return vocab_size,dataset


    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
   # embedding_matrix = {}
    # load glove vectors
    '''
    glove_dir = '/content/drive/MyDrive/Image_captioning/'
    f = open(os.path.join(glove_dir,'glove.6B.200d.txt'),encoding='utf-8')
    for line in f:
      values = line.split() # split each line into values - 0 index and remaining values
      word = values[0]
      coeff_ = np.asarray(values[1:],dtype='float32')
      embedding_matrix[word] = coeff_
    f.close()
    # For unique vocabulary words, find embeddings of words
    #embed_dim = 200
    #embedding_matrix_vocab = np.zeros((vocab_size,embed_dim))
    #for word,idx in dataset.vocab.stoi.items():
    #  embedding_vector = embedding_matrix.get(word)
    #  if embedding_vector is not None:
    #    embedding_matrix_vocab[idx] = embedding_vector
'''

    #model.decoderrnn.embed.weight.data = torch.Tensor(embedding_matrix_vocab).to(device)
    #for param in model.decoderrnn.embed.parameters():
     # param.requires_grad = False

    for name, param in model.encodercnn.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            print(type(param.data),name)
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN
    if load_model:
        step = load_checkpoint(torch.load("checkpoints_image_captioning.pth_2.tar"), model, optimizer)

    #model.train()
    losses = list()
    for epoch in tqdm(range(num_epochs)):
        #print_examples(model, device, dataset)

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint,"checkpoints_image_captioning.pth_2.tar")
            #shutil.copy("my_checkpoint.pth.tar","/content/drive/MyDrive/image_captionate_pytorch")
            

        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):            
            
            imgs = imgs.to(device)
            # here we divide captions into two lists - captions_train and captions_target
            captions_train = captions[:,:captions.shape[1]-1].to(device)
            captions_target = captions[:,1:].to(device)
            
            # captions = captions.to(device)

            outputs = model(imgs, captions_train)
            loss = criterion(outputs.view(-1,vocab_size), captions_target.contiguous().view(-1))
             
            losses.append(loss.item())
            
            writer.add_scalar("Training Loss", loss.item(), global_step= step)

            step+= 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = sum(losses)/len(losses)
        print('Epoch {}/{}'.format(epoch,num_epochs), '\t', 'loss:{}'.format(avg_loss))

vocab_size,dataset = train(only_vocab=True)
embed_size = 512
hidden_size = 512
vocab_size = vocab_size 
print(vocab_size)
num_layers = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

from utils import load_model
import torch
from model import DecoderRNN
model,step = load_model(model,torch.load("checkpoints_image_captioning.pth.tar",map_location=torch.device('cpu')))
model.eval()
from PIL import Image

transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])


decoderrnn_sample = DecoderRNN(embed_size,hidden_size,vocab_size)
test_image = transform(Image.open("/content/drive/MyDrive/Image_captioning/test_images/horse.png").convert("RGB")).unsqueeze(0)
#print(model,step)
# encode image using encoder
model.encodercnn.eval()
decoderrnn_sample.eval()
encode_image_features = model.encodercnn(test_image.to(device)).unsqueeze(1) # test_image
print(decoderrnn_sample.sample(inputs=encode_image_features.cpu().detach(),vocabulary = dataset.vocab))
### TESTING







