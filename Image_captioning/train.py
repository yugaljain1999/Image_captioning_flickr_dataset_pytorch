# Extract dataloader from get_loader.py and model architecture from model.py
import math
import os
import torch
import torchvision
from torchvision import transforms
from get_loader import get_loader
from torch.utils.data import Sampler
#from model import CNNtoRNN # CNNtoRNN will combine features and embeddings of text and then get predicted output 
from torch.utils.tensorboard import SummaryWriter # this will write all runs in a specified folder
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from utils import load_checkpoint,save_checkpoint
from tqdm import tqdm
from model import EncoderCNN,DecoderRNN
#transforms = transforms.Compose([transforms.Resize((356,356)),transforms.RandomCrop((299,299)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])
dataset,loader = get_loader('Flicker8k_Dataset','captions.txt',transforms=transforms)
print(loader)
vocab_size = len(dataset.vocab)
emb_size = 512
hidden_dim = 512
epochs = 20
print_every = 100
save_every = 1
#writer = SummaryWriter('/runs/flicker') # save events in this folder

learning_rate = 3e-4
save_model = True
load_model = False
Train = True
step = 0 # while saving checkpoints it will save each checkpoint

if torch.cuda.is_available():
    device = torch.device('cuda')
# define model
encoder = EncoderCNN(emb_size)
decoder = DecoderRNN(emb_size,hidden_dim,vocab_size)
encoder.to(device)
decoder.to(device) 
#model = CNNtoRNN(emb_size,hidden_dim,vocab_size,num_layers=1).to(device)
criteria = torch.nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi['<PAD>']) # ignore padded index while calculating loss
parameters = list(encoder.inceptionv3.fc.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters,lr = learning_rate)
# total steps
total_steps = math.ceil(len(dataset.all_caption_lengths) / dataset.batch_size )

'''
for name,param in model.encodercnn.inception.named_parameters():
    if "weight" in name or "bias" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
'''

#if load_model:
#    step = load_checkpoint(torch.load('checkpoints_image_captioning.pth.tar'),model,optimizer) # If you save checkpoints then pass model architecture as well 
    # here step refers to last step of saved checkpoints 
#model.train()
#losses = []
class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, always in the same order.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.num_samples = len(indices)
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples
# Load pretrained model weights before resuming trainng


for epoch in range(0,epochs):
    #if save_model:
    #    checkpoint = {'state_dict' : model.state_dict(),
    #                  'optimizer' : optimizer.state_dict(),
    #                  'step':step
                    
    #    save_checkpoint(checkpoint,'checkpoints_image_captioning.pth.tar')
    
    for i_step in range(1,total_steps+1):
      # Sampler is basically used to consider particular batch of classes associated with inputs
      # Suppose batch_size = 32 and m(number of samples-should be multiple of 32) = 4
      # Number of classes = 8 - so there will be 4 samples of each class
      indices = dataset.get_train_indices()
      new_sampler = SubsetSequentialSampler(indices = indices)
      loader.batch_sampler.sampler = new_sampler

      imgs,captions = next(iter(loader))

      imgs = imgs.to(device)
      captions = captions.to(device)
      #print('caption',captions.size())
      # make gradients of encoder and decoder to zero
      decoder.zero_grad()
      encoder.zero_grad()
      features = encoder(imgs)
      outputs = decoder(features,captions)
      #outputs = model(imgs,captions) # don't include last index
      #print('captions',captions.size())
      #print('outputs',outputs.size())
      loss = criteria(outputs.view(-1,vocab_size),captions.view(-1))
      #losses.append(loss.item())
      #writer.add_scalar("Training_loss",loss.item(),global_step=step) # save loss in each step for batches of images and captions
      
      loss.backward()
      optimizer.step()
      stats = 'Epoch {}/{} \t , step:{} \t , loss:{}'.format(epoch,epochs,i_step,loss.item())
      #step+=1
      print('\r' + stats)
    # avg_loss = sum(losses)/len(losses)
    # let's try to save model in pickle format instead of pth format
    if epoch % save_every == 0:
      torch.save(decoder.state_dict(),os.path.join('/content/drive/MyDrive/Image_captioning/models','decoder-{}.pkl'.format(epoch)))
      torch.save(encoder.state_dict(),os.path.join('/content/drive/MyDrive/Image_captioning/models','encoder-{}.pkl'.format(epoch)))
      torch.save(optimizer.state_dict(),os.path.join('/content/drive/MyDrive/Image_captioning/models','optim-{}.pkl'.format(epoch)))












