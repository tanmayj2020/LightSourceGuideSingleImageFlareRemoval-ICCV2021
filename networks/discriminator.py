# Required imports
import torch 
from torch import nn 
from networks.network_utils import BaseNetwork



class PatchDiscriminator(BaseNetwork):
    """
    Implements PatchGAN discriminator 

    Parameters
    ---------
    input_ch : int
            Input channels to the discriminator 
    opt :   
            User specififed arguments
    
    Attributes 
    ---------
    n_df : int 
            Discriminator features after first layer
    norm : 
            Type of norm to use
    activation : 
            Type of activation to use
    """
    def __init__(self , opt , input_ch ):
        super().__init__()
        # Required attributes
        activation = nn.LeakyReLU(0.2 , inplace=True)
        n_df = opt.n_df 
        norm = nn.InstanceNorm2d
        # Block for model
        block= []
        block+= [nn.Conv2d(input_ch , n_df , kernel_size = 4 , stride = 2 , padding =1) , activation]
        block+= [nn.Conv2d(n_df , n_df *2 , kernel_size = 4, stride =2 , padding = 1) , norm(n_df * 2) , activation]
        block+= [nn.Conv2d(n_df * 2, n_df * 4 , kernel_size = 4  , padding = 1 , stride = 2) ,norm(n_df * 4) , activation]
        block+= [nn.Conv2d(n_df*4 , n_df * 8 , kernel_size = 4 , stride = 1 , padding = 1) , norm(n_df * 8) , activation]
        block+= [nn.Conv2d(n_df*8 , 1 ,kernel_size=4 , stride = 1 , padding = 1)]
        # Dereferencing block
        self.block = nn.Sequential(*block)
        # Defining activation
        self.activation = torch.nn.Sigmoid()
        self.init_weights(opt.init_mean , opt.init_std)
        self.print_network()

    def forward(self , x):
        """
        Implements forward pass        
        Input 
        -----
        x : torch.tensor
            Concatenated output of CGenerator and Input  , Shape = (batch_size , channel , height , width)       
        Returns
        -------
        torch.tensor 
            Intermediate feature volume with 70*70 input receptive field , Shape = (batch_size , 1 , height' , width')
        """
        return self.activation(self.block(x))
