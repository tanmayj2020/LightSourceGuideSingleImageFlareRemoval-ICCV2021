# Required Imports
import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    Implements Residual Block
    
    Parameters 
    ----------
    n_channels : int 
                Number of channels(Input and Output)
    pad : 
                Type of padding to choose
    norm_layer : 
                Type of normalisation to choose
    activation : 
                Activation function to choose
    use_dropout : Bool
                Use Dropout or not

    Attributes 
    ----------
    block : List 
                List of all Modules to use 
    """

    def __init__(self , n_channels , pad, norm_layer , activation , use_dropout = False):
        """
        Initializes the block 
        """
        super().__init__()
        # Input : (batch_size , channels , height , width) :: Output : (batch_size , channels, height , width)
        block = [pad(1) , nn.Conv2d(n_channels, n_channels , kernel_size = 3 , padding = 0 , stride = 1) , norm_layer(n_channels) , activation]
        # Use Dropout 
        if use_dropout:
            block += [nn.Dropout(0.5)]
        # Input : (batch_size , channels , height , width) :: Output : (batch_size , channels,  height , width)
        block += [pad(1) , nn.Conv2d(n_channels , n_channels , kernel_size = 3 , padding = 9 , stride = 1) , norm_layer(n_channels)]
        # Deferencing 
        self.block = nn.Sequential(*block)

    def forward(self , x):
        """
        Implements the forward method 
        
        Parameters
        ----------
        x : torch.tensor
                Input feature volume , Shape = (batch_size , channel , height , widht)
        Returns 
        -------
        torch.Tensor 
                Output feature volume after residual block  , Shape = (batch_size , channel , height , widht)
        """
        return x + self.block(x)
