################ Load and Save checkpoints of a model ############
import torch
import torchvision
def load_checkpoint(checkpoints,model,optimizer):
    model.load_state_dict(checkpoints['state_dict'])
    optimizer = model.load_state_dict(checkpoints['optimizer'])
    step = checkpoints['step']
    return step
def save_checkpoint(checkpoints,filename):
    torch.save(checkpoints,'checkpoints_image_captioning.pth.tar')

def load_model(model,checkpoints):
    model.load_state_dict(checkpoints['state_dict'])
    step = checkpoints['step']
    return model,step
