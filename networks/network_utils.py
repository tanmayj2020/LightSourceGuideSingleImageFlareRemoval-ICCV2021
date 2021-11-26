from torch import nn
from functools import partial
from torch.nn import init


class BaseNetwork(nn.Module):

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params = param.numel()
        print("Network [{}] was created. Total number of parameters: {:.1f} million. "
            "To see the architecture, do print(network).".format(self.__class__.__name__, num_params / 1000000))
    
    def init_weights(self , mean = 0 , std = 0.02):
        
        def init_func(m):
            init.normal_(m.weight.data , mean , std)
            init.constant_(m.bias.data , 0)
        
        self.apply(init_func)
    
    def forward(self , *inputs):
        pass




#------------------------------
# Sets the padding method for the input
def get_pad_layer(type):
    # Chooses reflection , places mirror around boundary and reflects the value
    if type == "reflection":
        layer = nn.ReflectionPad2d
    # Replicates the padded area with nearest boundary value
    elif type == "replication":
        layer = nn.ReplicationPad2d
    # Padding of Image with constat 0 
    elif type == "zero":
        layer = nn.ZeroPad2d
    else:
        raise NotImplementedError("Padding type {} is not valid . Please choose among ['reflection' ,'replication' ,'zero']".format(type))
    
    return layer

    
#----------------------------------
# Sets the norm layer 
def get_norm_layer(type):
    if type == "BatchNorm2d":
        layer = partial(nn.BatchNorm2d , affine = True) 
    elif type == "InstanceNorm2d":
        layer = partial(nn.InstanceNorm2d ,affine = False)
    else : 
        raise NotImplementedError("Norm type {} is not valid. Please choose ['BatchNorm2d' , 'InstanceNorm2d']".format(type))
    
    return layer

    
