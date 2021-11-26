# Necessary imports
from torch import nn
import torch
from torch.utils import data

from os import path , listdir
from PIL import Image


class Dataset(data.Dataset):
    """
    Implements CustomDataset
    
    Parameters
    ----------
    opt : 
        User specified arguments
    transforms : 
        Transform to be applied on input data
    
    Atrributes
    ----------
    data_path :
            Path to the dataset
    flare_images :
            List of flare images in dataset
    flare_free_images :
            List of flare free images in dataset    
    """

    def __init__(self , opt , transforms):
        """
        Initialising the dataset       
        """
        # Path to the dataset
        self.data_path = opt.data_path
        # Path to flare anf flare free images
        self.flare_free_path = path.join(self.data_path , "flare_free")
        self.flare_path = path.join(self.data_path , "flare")
        # List of flare anf flare free images
        self.flare_free_images = listdir(self.flare_free_path)
        self.flare_images = listdir(self.flare_path)
        # Length of flare images , flare free images and dataset length
        self.flare_length = len(self.flare_images)
        self.flare_free_length = len(self.flare_free_images)
        self.dataset_length = max(self.flare_length ,  self.flare_free_length)
        # Transforms to be applied to the images
        self.transform = transforms

    def __getitem__(self, index):
        """
        Implements get items
        Parameters
        ----------
        index : int
                Index of the dataset
        Returns 
        -------
        torch.tensor 
                Tensor for flare images and flare free images
        """
        #Getting image paths
        flare_image_index = self.flare_images[index % self.flare_length]
        flare_free_image_index = self.flare_free_images[index % self.flare_free_length]
        
        flare_image_path = path.join(self.flare_path , flare_image_index)
        flare_free_image_path = path.join(self.flare_free_path , flare_free_image_index)

        # Loading image in memory 
        flare_image = Image.open(flare_image_path).convert("RGB")
        flare_free_image = Image.open(flare_free_image_path).convert("RGB")
        
        # Aplying transformations to images
        flare_image = self.transform(flare_image)
        flare_free_image = self.transform(flare_free_image)

        return flare_image ,flare_free_image

    def __len__(self):
        """
        Calculates length of the dataset
        """
        return self.dataset_length
    

