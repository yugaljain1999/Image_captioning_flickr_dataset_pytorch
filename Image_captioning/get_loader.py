import torchvision
import torch
import os
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from torchvision import datasets,transforms # here transforms function is used to transform images into fixed size, cropping
import spacy
# Load flicker dataset
# Build vocabulary and numercalize captions
# Create batches of images and captions - > images should be transformed using transforms function in pytorch
# Note: Update collate function while using data loader function to set padded index of captions
spacy_tok = spacy.load('en')
class Vocabulary:

    def __init__(self,word_threshold):
        self.word_threshold = word_threshold
        self.stoi = {'<PAD>':0,'<SOS>':1,'<EOS>':2,'<UNK>':3} # till now there are just special tokens in vocab dictionary, in build vocab function we will add indexes of all tokens of captions
        self.tois = {0:'<PAD>',1:'<SOS>',2:'<EOS>',3:'<UNK>'}
    # define static method tokenizer - compute without taking any reference from instance variables of obhect of that class
    def __len__(self):
        return len(self.tois)
    
    @staticmethod
    def tokenizer(text):
        return [tok.text.lower() for tok in spacy_tok.tokenizer(text)]

    def build_vocab(self,sentences_list):
        word_count = {}
        idx = 4 # because 0-3 indexes are for four special tokens-<PAD>,<SOS>,<EOS>,<UNK>
        for sent in sentences_list:
            for word in self.tokenizer(sent):
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word]+=1
                if word_count[word] == self.word_threshold: # if any word occurs more than or equal to value of word_threshold times then add that word into dictionary otherwise not
                    self.stoi[word] = idx  # __init__ function variable named self.stoi modified here
                    self.tois[idx] = word
                    idx += 1 # idx is incrementing by 1 every time to store those words which counts 5 times atmost
        
    def numercalize_sentences(self,text): # remember self is used just because to use __init__ function variables
        tokenized_text = self.tokenizer(text)
        #print('tokenized_text',tokenized_text)
        #print('self.stoi',self.stoi)
        # convert each token to index using stoi dictionary 
        return [self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in tokenized_text]

class FlickerDataset(Dataset):
    def __init__(self,root_dir,caption_file,transforms,word_threshold=5): # root_dir - directory of images , annotation_file - captions.txt
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)

        self.transforms = transforms
        self.batch_size = 128
        self.images = self.df['image'][:10000] # images filenames
        self.captions = self.df['caption'][:10000]
        self.vocab = Vocabulary(word_threshold=word_threshold)
        self.vocab.build_vocab(self.captions.tolist()) # here captions are in list format
        self.all_caption_lengths = []
        for i in range(len(self.captions.tolist())):
          self.all_caption_lengths.append(len(self.vocab.numercalize_sentences(self.captions.tolist()[i])))
    def __len__(self):
        return len(self.df)

    def __getitem__(self,index): # Customizing getitem function for our usecase
        image_id = self.images[index] # here index - index of each image 
        caption = self.captions[index]
        image = Image.open(os.path.join(self.root_dir,image_id)).convert('RGB')
        
        ######## Preprocess image and caption and return ###########
        if self.transforms is not None:
            image = self.transforms(image)
        numerical_caption = self.vocab.numercalize_sentences(caption) # here caption is just single instance of captions dataframe
        # add indexes of <SOS> and <EOS> special tokens
        numerical_token = [self.vocab.stoi['<SOS>']]
        numerical_token+=numerical_caption # adding two lists
        numerical_token.append(self.vocab.stoi['<EOS>'])
        return image,torch.tensor(numerical_token)         # torch.Tensor has limited functionality as compare to torch.tensor

    def get_train_indices(self):
      # all caption lengths
      all_caption_lengths = []
      for i in range(len(self.captions.tolist())):
        all_caption_lengths.append(len(self.vocab.numercalize_sentences(self.captions.tolist()[i]))) # here for each caption - length of numericalized sentences are saved
      sel_length = np.random.choice(all_caption_lengths)
      all_indices = np.where([all_caption_lengths[i] == sel_length for i in np.arange(len(all_caption_lengths))])[0]
      # indices
      indices = list(np.random.choice(all_indices,size = self.batch_size))
      return indices

##### collate function class which is a parameter for DataLoader function of torch.utils.data.DataLoader to modify for own data
class collate:
    def __init__(self,pad_idx):
        self.pad_idx = pad_idx
    def __call__(self,batch): # collate class call this function for batching but here we need to pad data
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images,dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets,batch_first=True,padding_value=self.pad_idx)
        return images,targets


# define get_loader function to load whole preprocessed data using DataLoader 
def get_loader(root_dir,caption_file,transforms,batch_size=128,pin_memory=True,shuffle=True):
    dataset = FlickerDataset(root_dir,caption_file,transforms)
    pad_idx = dataset.vocab.stoi['<PAD>']
    loader = DataLoader(dataset,batch_size=batch_size,pin_memory=pin_memory,shuffle=shuffle,num_workers=5,collate_fn=collate(pad_idx=pad_idx))
    return dataset,loader


        


if __name__ == '__main__':
    transforms = transforms.Compose([transforms.Resize((356,356)),transforms.RandomCrop((299,299)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    dataset,loader = get_loader('Flicker8k_Dataset','captions.txt',transforms = transforms)
    for id,(img,caption) in enumerate(loader):
        #print(caption)
        #print(img)
        break

                    

