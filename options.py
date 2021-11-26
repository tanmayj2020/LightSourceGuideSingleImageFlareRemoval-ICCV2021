import os 
import argparse
from random import choices 
import wandb
import sys


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_wandb" , action="store_true" , help ="Use wandb logger")
    parser.add_argument("--wandbrunname" , required = "--use_wandb" in sys.argv ,help="WandBias run name")
    parser.add_argument("--projectname" , required = "--use_wandb" in sys.argv ,help="WandBias Project Name")
    parser.add_argument("--wandbusername" , required = "--use_wandb" in sys.argv ,help="WandBias UserName")
    parser.add_argument("--seed" , type = int , default =3407  ,help="Random Seed")
    parser.add_argument("--train_mode",type=str ,choices= ["flare_free" , "flare"] , required=True , help="Train either 'flare' or 'flare_free' images")
    parser.add_argument("--learning_rate" , default = 2e-4 , type=float , help="Initial Learning Rate")
    parser.add_argument("--n_epochs" , default= 100, type=int , help="Number of epochs")
    parser.add_argument("--b1" , default = 0.5 , type=float , help="Adam B1")
    parser.add_argument("--b2" , default = 0.9999 , type=float , help="Adam B2")
    parser.add_argument("--batch_size" , default = 16 , type=int , help="Batch Size")
    parser.add_argument("--load_height" , default = 512 , type=int , help="Height of Input Image")
    parser.add_argument("--load_width" , default = 512 , type=int , help="Width of Input Image")
    parser.add_argument("--shuffle" , action="store_true",  help="Shuffle dataset")
    parser.add_argument('--log_interval', type=int, default=20, metavar="N", help="Log per N steps in a Epoch")
    parser.add_argument('--use_schedular', action= "store_true", help="Use lr schedular")
    parser.add_argument('--decay_rate', default = 0.1 , type = float , required="--use_schedular" in sys.argv, help="Decay rate of exponential schedular")
    parser.add_argument('--checkpoint_dir', type=str, help="Directory to save checkpoints" , default= "./checkpoints")
    parser.add_argument("--load_model" , action="store_true" , help="Load pretrained model")
    parser.add_argument('--load_optimizer', action= "store_true", help="Load optimizer (Only works if loading the model)")
    parser.add_argument('--n_gf', type=int , default = "64", help="Feature maps in first generator layer")
    parser.add_argument('--norm_type', type=str , default = "BatchNorm2d", help="Norm used in Generator" , choices = ["BatchNorm2d" , "InstanceNorm2d"])
    parser.add_argument('--pad_type', type=str , default = "reflection", help="Padding used in Generator" , choices = ["reflection" , "replication" , "zero"])
    parser.add_argument('--n_downsample', type=int , default = 2, help="Number of time downsample or Upsample in Generator")
    parser.add_argument('--n_residual', type=int , default = 9, help="Number of Residual Block in Generator")
    parser.add_argument('--init_mean', type=float , default = 0.0, help="Mean of Gaussian weight initialisation")
    parser.add_argument('--init_std', type=float, default = 0.02, help="Standard deviation of Guassian weight intialisation")
    parser.add_argument('--n_edf', type=int , default = "16", help="Feature maps in first encoder-decoder layer")
    parser.add_argument('--n_df', type=int , default = "64", help="Feature maps in first Discriminator layer")
    parser.add_argument('--data_path', type=str, help="Dataset Root Directory" , default= "./trainDataset")

    args = parser.parse_args()
    return args
    







    











