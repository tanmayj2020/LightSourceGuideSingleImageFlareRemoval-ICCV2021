from torchvision import transforms
from torch import nn
import torchvision
import torch 
import numpy as np
import random
import os

def get_transforms(height , width):

    transform = transforms.Compose(
        [
            transforms.Resize((height , width)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor() , 
            transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))
        ]
    )
    return transform


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.model =torchvision.models.vgg16(pretrained=True).features[:30].eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self , x):
        return self.model(x)

def adjust_learning_rate(optim ,epoch , initial_lr , decay_rate):
    lr = initial_lr * torch.exp(-decay_rate * epoch)
    for param_group in optim.param_groups:
        param_group["lr"] = lr
    

def seed_everything(seed = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


