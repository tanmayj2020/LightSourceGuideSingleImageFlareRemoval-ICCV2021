# Required imports
from torch import nn
import torch

from networks.network_utils import BaseNetwork



class EncoderDecoder(BaseNetwork):
    """
    Implements Encoder Decoder 

    Parameters
    ----------
    opt : 
        User specified arguments
    input_ch : int 
        Number of input channels 
    output_ch : int 
        Number of output channels 
    Attributes 
    ----------
    n_edf : int 
        Number of channels after first conv layer
    norm : 
        Type of norm to use
    activation : 
        Type of activation to use
    model : 
        List containing all architecture modules 

    """    
    def __init__(self , opt , input_ch , output_ch):
        """
        Initializes the module architecture
        """
        n_edf = opt.n_edf
        norm = nn.InstanceNorm2d
        activation = nn.ReLU(inplace=True)

        model = []
        model += [nn.Conv2d(input_ch , n_edf , kernel_size=7 , stride = 1 , padding = 3) , norm(n_edf) , activation]
        model += [nn.Conv2d(n_edf , n_edf * 4 , kernel_size= 3 , padding = 1 , stride= 2) , norm(n_edf * 4) , activation]
        model += [nn.Conv2d(n_edf * 4 , n_edf * 8 , kernel_size= 3 , padding = 1 , stride= 2) , norm(n_edf * 8) , activation]
        model += [nn.Conv2d(n_edf * 8 , n_edf * 16 , kernel_size= 3 ,padding = 1 , stride = 1), norm(n_edf * 16) , activation]
        model += [nn.Upsample(scale_factor=2 , mode="nearest") , nn.Conv2d(n_edf * 16 , n_edf * 8 , kernel_size=3 , stide = 1 , padding =1) , norm(n_edf * 8) , activation]
        model += [nn.Upsample(scale_factor=2 , mode="nearest") , nn.Conv2d(n_edf * 8 , n_edf * 4 , kernel_size=3 , stide = 1 , padding =1) , norm(n_edf * 4) , activation]
        model += [nn.Conv2d(n_edf * 4 , n_edf * 2, kernel_size= 3 ,padding = 1 , stride = 1), norm(n_edf * 2) , activation]
        model += [nn.Conv2d(n_edf * 2 , n_edf, kernel_size= 3 ,padding = 1 , stride = 1), norm(n_edf) , activation]
        model += [nn.Conv2d(n_edf , output_ch, kernel_size= 7 ,padding = 3 , stride = 1)]

        self.model = nn.Sequential(*model)        
        self.activation = torch.nn.Sigmoid()

        self.init_weights(opt.init_mean , opt.init_std)
        self.print_network()

    def forward(self , x):
        """
        Implements forward pass

        Input 
        -----
        x : torch.tensor 
            Input Image tensor , shape = (batch_size , input_ch , height , width)
        Returns
        -------
        torch.tensor 
            Output feature volume , shape = (batch_size , output_ch , height , width)
        """
        return self.activation(self.model(x))



