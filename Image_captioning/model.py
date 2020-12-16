import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics
import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True) # trying resnet50
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
'''
class EncoderCNN(nn.Module):
  def __init__(self,embed_size):
    super(EncoderCNN,self).__init__()
    self.inceptionv3 = models.inception_v3(pretrained=True,aux_logits=False)
    for param in self.inceptionv3.parameters():
      param.requires_grad_(False)
    #modules = list(inceptionv3.children())[:-1]
    #self.inceptionv3 = nn.Sequential(*modules)
    self.inceptionv3.fc = nn.Linear(self.inceptionv3.fc.in_features,embed_size)

  def forward(self,image):
    features = self.inceptionv3(image)
    return features




class DecoderRNN(nn.Module):
  def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
      super(DecoderRNN, self).__init__()
      self.hidden_size = hidden_size
      self.embed_size = embed_size
      self.vocab_size = vocab_size
      self.device = device
      self.embed = nn.Embedding(vocab_size, embed_size) # embedding layer 
      self.lstm = nn.LSTM(embed_size, hidden_size,batch_first=True,num_layers = num_layers,dropout=0)
      self.linear = nn.Linear(hidden_size, vocab_size)
      self.dropout = nn.Dropout(0.5)
  
  # initialize hidden states
  def init_hidden(self,batch_size):
    return (torch.zeros(1,batch_size,self.hidden_size,device=device),torch.zeros(1,batch_size,self.hidden_size,device=device))


  def forward(self, features, captions):
      batch_size = features.size(0)
      self.hidden = self.init_hidden(batch_size) 
      # INITIALIZE HIDDEN AND CELL STATES to zero
      captions_embed = self.embed(captions[:,:-1])
      captions_embed = torch.cat((features.unsqueeze(1),captions_embed),dim=1)
      #print('captions_embed_shape',captions_embed.shape)
      # change initial hidden state using prior output
      lstm_out,self.hidden = self.lstm(captions_embed,self.hidden)
      out = self.linear(lstm_out)
      return out

  def sample(self,inputs,vocabulary,max_len=20):
    # here inputs are captions
    hidden = (torch.zeros(1,inputs.shape[0],self.hidden_size),torch.zeros(1,inputs.shape[0],self.hidden_size))
    #hidden = self.init_hidden(inputs.shape[0]).cpu().detach()
    out_list = []
    word_len = 0
    with torch.no_grad():
      while word_len<max_len:
        lstm_out,hidden = self.lstm(inputs,hidden)
        out = self.linear(lstm_out)
        out = out.squeeze(1) # dimension reduction
        out = out.argmax(dim=1) # get indices of maximum values in increasing order
        #print('out.shape',out.shape)
        out_list.append(out.item())
        # change inputs 
        inputs = self.embed(out.unsqueeze(0))  # here remember one thing - initially we squeeze output on dim=0 now during unsqueezing we have to unsqueeze on dim = 1
        word_len+=1
        if out == 1:
          break
      # convert indices to tokens
      out_caption = [vocabulary.tois[out] for out in out_list] # out_list is a list of indices of tokens(captions)
    return " ".join(out_caption)




class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        # print("Model Specs")
        # print("Embed size : ", embed_size)
        # print("Hidden size : ", hidden_size)
        # print("Vocab size : ", vocab_size)
        self.encodercnn = EncoderCNN(embed_size)
        self.decoderrnn = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers=1)


    def forward(self, images, captions):
        features = self.encodercnn(images)
        outputs = self.decoderrnn(features, captions) # here captions specifies some part of string as captions_train and captions_target wouild be ground truth
        return outputs

    
    ### TESTING
    def caption_image(self, image, vocabulary, max_length=50):
      result_caption = []

      with torch.no_grad():
          # print("Image Shape : ", image.shape)
          # print(self.encoderCNN(image).shape)
          self.encodercnn.eval()
          self.decoderrnn.eval()
          
          x = self.encodercnn(image) # encoded image
          #print('image_size',x.size())
          hidden_state = torch.zeros((x.size(0),self.decoderrnn.hidden_size)).cuda()
          cell_state = torch.zeros((x.size(0),self.decoderrnn.hidden_size)).cuda()
          #print('hidden_state',hidden_state.size())
          for _ in range(max_length): # maximum length of image
              # print(x.shape)
              hidden_state,cell_state = self.decoderrnn.lstmcell(x, (hidden_state,cell_state))
              #print('_',_)
              output = self.decoderrnn.linear(hidden_state)
              #print('hidden_state',hidden_state)
              predicted = output.argmax(1)
              #print('output',output.size())
              #print('predicted',predicted.size())
              #print('predicted_item',predicted.item())
              result_caption.append(predicted.item())
              x = self.decoderrnn.embed(predicted).squeeze(1)
              #print('updated_x',x.size())
              if vocabulary.tois[predicted.item()] == "<EOS>":
                  break

          return ' '.join([vocabulary.tois[idx] for idx in result_caption])


        


        


