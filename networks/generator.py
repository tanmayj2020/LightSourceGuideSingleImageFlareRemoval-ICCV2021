# Required imports
import torch 
from torch import nn
# Custom Imports
from networks.network_utils import BaseNetwork, get_norm_layer , get_pad_layer
from networks.res_block import ResidualBlock



class Generator(BaseNetwork):
    """
    Implements Generator

    Parameters
    ----------
    opt : 
        User specified arguments
    input_ch : int 
        Number of channels in the input
    output_ch : int 
        Number of channels in Output
    
    Attributes 
    ----------
    model : 
            List of all network modules
    n_gf : int 
            Number of output channel of first convolution of gen
    pad : 
            Type of padding to use
    norm : 
            Type of norm to use
    """
    def __init__(self, opt , input_ch , output_ch):
        """
        Initiates the generator
        """
        super().__init__()

        activation = nn.ReLU()
        n_gf = opt.n_gf
        norm = get_norm_layer(opt.norm_type)
        pad = get_pad_layer(opt.pad_type)

        model = []
        # Same shape as input 
        model += [pad(3) , nn.Conv2d(input_ch , n_gf , kernel_size = 7 , padding = 0) , norm(n_gf) , activation]
        # Downsampling the input
        for num in range(opt.n_downsample):
            model += [nn.Conv2d(n_gf , 2 * n_gf , kernel_size = 3 , padding =1 ,  stride = 2) , norm(n_gf * 2) , activation]
            n_gf = n_gf * 2
        # Same shape as the input
        for num in range(opt.n_residual):
            model += [ResidualBlock(n_gf , pad , norm , activation)]
        # Upsampling the input
        for num in range(opt.n_downsample):
            model += [nn.Upsample(scale_factor = 2 , mode="nearest") ,nn.Conv2d(n_gf ,n_gf // 2 , kernel_size = 3 , padding = 1 , stride = 1) ,norm(n_gf // 2) , activation ]
            n_gf = n_gf //2
        # Same shape as the input 
        model += [pad(3) , nn.Conv2d(n_gf , output_ch , kernel_size = 7 , stride = 1 , padding = 0)]
        self.model = nn.Sequential(*model)
        #Printing the model
        self.init_weights(opt.init_mean , opt.init_std)
        self.print_network()
    

    def forward(self , x):
        """
        Implements the forwad pass

        Input
        -----
        x : torch.tensor 
            Input feature volume : Shape = (batch_size , input_ch , height , width)
        Returns
        -------
        Output image : Shape = (batch_size , output_ch , height , widht)
        """
        return self.model(x)

