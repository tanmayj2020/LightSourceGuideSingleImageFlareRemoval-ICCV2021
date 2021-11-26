import torch 
from torch import nn
from torch.nn.modules import loss
import torch.optim as optim
from torch.random import seed
from torch.utils.data.dataloader import DataLoader
import config
from networks import PatchDiscriminator, Generator
from networks import EncoderDecoder
from dataset import Dataset
from torch.utils import data
import utils
import os
from tqdm import tqdm
from options import get_args
import time
import wandb
import torchvision.transforms.functional as F


# TODO : Check entire training code , ResidualBlock from nn.Module or BaseNetwor, Options , Comments , Testing Scripts , PaperConcepts of pretraining checkout
# TODO : ReadPaperComplete(Issue with light source mask as groundtruth) , Preprocess the dataset , Saving of images in utils.py



class TrainFlareFree:

    def __init__(self , opt):
        print("=> Training for flare free images")
        self.opt = opt
        self.initial_lr = opt.learning_rate
        self.n_epoch = opt.n_epochs
        self.b1 = opt.b1
        self.b2 = opt.b2
        self.batch_size = opt.batch_size

        self.load_height = opt.load_height 
        self.load_width = opt.load_width

        self.device = config.DEVICE
        self.FR = Generator(opt, 4 , 3).to(self.device)
        self.FG = Generator(opt,4 , 3).to(self.device)
        self.LSD = EncoderDecoder(opt , 3 , 1).to(self.device)
        self.FD = EncoderDecoder(opt , 3 , 1).to(self.device)
        self.FGD = PatchDiscriminator(opt , 3).to(self.device)
        
        self.opt_disc_flare_generator = optim.Adam(self.FGD.parameters() , lr = self.initial_lr , betas=(self.b1 , self.b2))
        self.opt_flare_free = optim.Adam(list(self.FR.parameters()) + list(self.FG.parameters()) + list(self.LSD.parameters()) + list(self.FD.parameters()), lr = self.initial_lr , betas=(self.b1 , self.b2))

        self.transforms = utils.get_transforms(self.load_height , self.load_width)

        self.dataset = Dataset(opt, self.transforms)
        self.dataloader = data.DataLoader(self.dataset , batch_size=self.batch_size , shuffle = opt.shuffle ,num_workers=config.NUM_WORKERS , pin_memory=True)

        self.feature_extractor = utils.VGG().to(self.device)

        self.network_scaler = torch.cuda.amp.GradScaler()
        self.discriminator_scaler = torch.cuda.amp.GradScaler()

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def train(self):
        for epoch in range(self.n_epoch):
            start_time = time.time()
            running_loss = 0
            counter = 0
            data  = []
            loop = tqdm(self.dataloader , leave=True , desc=f"Epoch {epoch:>2}")
            for batch_idx , (flare, flare_free) in enumerate(loop):
                flare = flare.to(config.DEVICE)
                flare_free = flare_free.to(config.DEVICE)

                #Training discriminator
                with torch.cuda.amp.autocast():
                    light_source_mask_real = self.LSD(flare_free)
                    fg_input = torch.cat((light_source_mask_real , flare_free) , 1)
                    fake_flare_image = self.FG(fg_input)
                    # Add detach here with proper reason
                    discriminator_fake = self.FGD(fake_flare_image)
                    discriminator_real = self.FGD(flare)
                    discriminator_fake_loss = self.mse(discriminator_fake , torch.zeros_like(discriminator_fake))
                    discriminator_real_loss = self.mse(discriminator_real , torch.ones_like(discriminator_real))
                    total_discriminator_loss = discriminator_fake_loss + discriminator_real_loss

                self.opt_disc_flare_generator.zero_grad()
                self.discriminator_scaler.scale(total_discriminator_loss).backward()
                self.discriminator_scaler.step(self.opt_disc_flare_generator)
                self.discriminator_scaler.update()

                # Training entire module
                with torch.cuda.amp.autocast():
                    light_source_mask_real = self.LSD(flare_free)
                    fg_input = torch.cat((light_source_mask_real , flare_free) , 1)
                    fg_output = self.FG(fg_input)
                    flare_mask_fake = self.FD(fg_output)
                    flare_removal_input = torch.cat((flare_mask_fake , fg_output) , 1)
                    reconstructed_flare_free_image = self.FR(flare_removal_input)
                    light_source_mask_fake = self.LSD(reconstructed_flare_free_image)
                    discriminator_fake_output= self.FGD(fg_output)
                    generator_adversial_loss = self.mse(discriminator_fake_output , torch.ones_like(discriminator_fake_output))
                    light_source_loss = self.l1(light_source_mask_real , light_source_mask_fake)
                    real_image_features = self.feature_extractor(flare_free)
                    fake_image_features = self.feature_extractor(reconstructed_flare_free_image)
                    cycle_loss = self.l1(flare_free , reconstructed_flare_free_image) + self.l1(real_image_features , fake_image_features)
                    total_loss = config.WEIGHT5 * generator_adversial_loss + config.WEIGHT6 * light_source_loss + config.WEIGHT7 * cycle_loss
                    running_loss += total_loss.item()
                
                self.opt_flare_free.zero_grad()
                self.network_scaler.scale(total_loss).backward()
                self.network_scaler.step(self.opt_flare_free)
                self.network_scaler.update()

                counter += 1
                if (batch_idx + 1) % self.opt.log_interval == 0:
                    info = {"totalLoss" : total_loss , "cycleLoss" : cycle_loss , "adversialLoss" : generator_adversial_loss, "lightSourceLoss" : light_source_loss}
                    if self.opt.use_wandb: 
                        wandb.log(info)
                        flare_free_image_log = F.to_pil_image(flare_free[0] * 0.5 + 0.5)        
                        light_source_mask_real_log = F.to_pil_image(light_source_mask_real[0] * 0.5 + 0.5)
                        fg_output_log = F.to_pil_image(fg_output[0] * 0.5 + 0.5)
                        reconstructed_flare_free_image_log = F.to_pil_image(reconstructed_flare_free_image[0] * 0.5 + 0.5)
                        data.append([batch_idx + 1 , wandb.Image(flare_free_image_log) , wandb.Image(light_source_mask_real_log), wandb.Image(fg_output_log), wandb.Image(reconstructed_flare_free_image_log)])

                    loop.set_postfix(info)
                
            print("Epoch : {} || Time elapsed {} || TotalLoss {}".format(epoch + 1 ,time.time()- start_time , float(running_loss/ counter)))
            if self.opt.use_wandb: 
                wandb.log(epoch + 1)
                columns = ["BatchIndex" , "FlareFreeImage",  "LightSourceMask" , "FlareGenerationOutput" , "ReconstructedImage"]
                table = wandb.Table(data=data , columns = columns)
                wandb.log({f"{self.opt.train_mode} => {epoch + 1} predictions" : table})


            if (epoch + 1 )% 100 == 0:
                if self.opt.use_schedular:
                    utils.adjust_learning_rate(self.opt_disc_flare_generator, epoch + 1 , self.initial_lr , self.opt.decay_rate)
                    utils.adjust_learning_rate(self.opt_flare_free, epoch + 1, self.initial_lr, self.opt.decay_rate)

            if (epoch + 1) % 5 == 0:
                self.save_model()




    def save_model(self):
        print("=> Saving Model")
        checkpoint_path = self.opt.checkpoint_dir
        torch.save(self.FR.state_dict() , os.path.join(checkpoint_path , "FR.pth"))
        torch.save(self.FG.state_dict() , os.path.join(checkpoint_path , "FG.pth"))
        torch.save(self.LSD.state_dict() , os.path.join(checkpoint_path , "LSD.pth"))
        torch.save(self.FD.state_dict() , os.path.join(checkpoint_path , "FD.pth"))
        torch.save(self.FGD.state_dict() , os.path.join(checkpoint_path , "FGD.pth"))
        torch.save(self.opt_disc_flare_generator.state_dict() , os.path.join(checkpoint_path , "opt_disc_flare_generator.pth"))
        torch.save(self.opt_flare_free.state_dict() , os.path.join(checkpoint_path , "opt_flare_free.pth"))
        print("[+] Weights saved.")

    def load_model(self):
        print("=> Loading Checkpoints")
        checkpoint_path = self.opt.checkpoint_path
        try:
            self.FR.load_state_dict(torch.load(os.path.join(checkpoint_path , "FR.pth") ,map_location = self.device))
            self.FG.load_state_dict(torch.load(os.path.join(checkpoint_path , "FG.pth") , map_location=self.device))
            self.LSD.load_state_dict(torch.load(os.path.join(checkpoint_path , "LSD.pth") , map_location=self.device))
            self.FD.load_state_dict(torch.load(os.path.join(checkpoint_path , "FD.pth") , map_location=self.device))
            self.FGD.load_state_dict(torch.load(os.path.join(checkpoint_path , "FGD.pth") , map_location=self.device))
            if self.opt.load_optimizer:
                self.opt_disc_flare_generator.load_state_dict(torch.load(os.path.join(checkpoint_path , "opt_disc_flare_generator.pth") , map_location=self.device))
                self.opt_flare_free.load_state_dict(torch.load(os.path.join(checkpoint_path , "opt_flare_free.pth") , map_location=self.device))
                self.initial_lr = self.opt_flare_free.param_groups[0]['lr']

            print("[+] Weights Loaded")
        except FileNotFoundError as e:
            print(f"[!] {e} , skipping loading weights")

class TrainFlare:

    def __init__(self , opt):
        print("=> Training for flare images")
        self.opt = opt
        self.initial_lr = opt.learning_rate
        self.n_epoch = opt.n_epoch 
        self.b1 = opt.b1 
        self.b2 = opt.b2
        self.batch_size = opt.batch_size


        self.load_height = opt.load_height 
        self.load_width = opt.load_width

        self.device = config.DEVICE
        self.FR = Generator(opt, 4 , 3).to(self.device)
        self.FG = Generator(opt,4 , 3).to(self.device)
        self.LSD = EncoderDecoder(opt , 3 , 1).to(self.device)
        self.FD = EncoderDecoder(opt , 3 , 1).to(self.device)
        self.FRD = PatchDiscriminator(opt , 3).to(self.device)

        self.opt_disc_flare_removal = optim.Adam(self.FRD.parameters() , lr =self.initial_lr , betas= (self.b1 , self.b2))
        self.opt_flare = optim.Adam(list(self.FR.parameters()) + list(self.FG.parameters()) + list(self.LSD.parameters()) + list(self.FD.parameters()) , lr= self.initial_lr , betas =(self.b1 , self.b2))

        self.transform = utils.get_transforms(self.load_height , self.load_width)

        self.dataset = Dataset(opt ,self.transform)
        self.dataloader = data.DataLoader(self.dataset , self.batch_size , shuffle = not opt.not_shuffle ,num_workers=config.NUM_WORKERS , pin_memory=True)
        
        self.feature_extractor = utils.VGG().to(self.device)


        self.network_scaler = torch.cuda.amp.GradScaler()
        self.discriminator_scaler = torch.cuda.amp.GradScaler()

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()



    def train(self):
        for epoch in range(self.n_epoch):
            running_loss = 0
            counter = 0 
            start_time = time.time()
            data = []
            loop = tqdm(self.dataloader , leave=True , desc=f"Epoch {epoch:>2}")
            for batch_idx , (flare, flare_free) in enumerate(loop):
                flare = flare.to(config.DEVICE)
                flare_free = flare_free.to(config.DEVICE)

                #Training discriminator
                with torch.cuda.amp.autocast():
                    light_source_mask_real = self.LSD(flare)
                    flare_mask_real = self.FD(flare)
                    fr_input = torch.cat((flare_mask_real , flare) , 1)
                    fake_flare_free_image = self.FR(fr_input)
                    # Add detach here with proper reason
                    discriminator_fake = self.FRD(fake_flare_free_image)
                    discriminator_real = self.FRD(flare_free)
                    discriminator_fake_loss = self.mse(discriminator_fake , torch.zeros_like(discriminator_fake))
                    discriminator_real_loss = self.mse(discriminator_real , torch.ones_like(discriminator_real))
                    total_discriminator_loss = discriminator_fake_loss + discriminator_real_loss


                self.opt_disc_flare_removal.zero_grad()
                self.discriminator_scaler.scale(loss).backward()
                self.discriminator_scaler.step(self.opt_disc_flare_removal)
                self.discriminator_scaler.update()

                # Training entire module 
                with torch.cuda.amp.autocast():
                    light_source_mask_real = self.LSD(flare)
                    flare_mask_real = self.FD(flare)
                    fr_input = torch.cat((flare_mask_real , flare) , 1)
                    fr_output = self.FR(fr_input)
                    light_source_mask_fake = self.LSD(fr_output)
                    flare_generation_input = torch.cat((light_source_mask_real , fr_output) , 1)
                    reconstructed_flare_image = self.FG(flare_generation_input)
                    flare_mask_fake = self.FD(reconstructed_flare_image)
                    discriminator_fake_output= self.FRD(fr_output)
                    generator_adversial_loss = self.mse(discriminator_fake_output , torch.ones_like(discriminator_fake_output))
                    light_source_loss = self.l1(light_source_mask_real , light_source_mask_fake)
                    real_image_features = self.feature_extractor(flare)
                    fake_image_features = self.feature_extractor(reconstructed_flare_image)
                    cycle_loss = self.l1(flare, reconstructed_flare_image) + self.l1(real_image_features , fake_image_features)
                    flare_loss = self.l1(flare_mask_real ,flare_mask_fake )
                    total_loss = config.WEIGHT2 * generator_adversial_loss + config.WEIGHT1 * light_source_loss + config.WEIGHT4 * cycle_loss + config.WIEGHT3 * flare_loss
                    running_loss += total_loss.item()

                self.opt_flare.zero_grad()
                self.network_scaler.scale(total_loss).backward()
                self.network_scaler.step(self.opt_flare)
                self.network_scaler.update()
                

                counter += 1            
                if (batch_idx + 1) % self.opt.log_interval == 0:
                    info = {"totalLoss" : total_loss  , "cycleLoss" : cycle_loss , "adversialLoss" : generator_adversial_loss, "lightSourceLoss" : light_source_loss  , "flareLoss" : flare_loss }
                    if self.opt.use_wandb: 
                        wandb.log(info)
                        flare_image_log = F.to_pil_image(flare[0]* 0.5 + 0.5)
                        flare_mask_real_log = F.to_pil_image(flare_mask_real[0]* 0.5 + 0.5)
                        light_source_mask_real_log = F.to_pil_image(light_source_mask_real[0]* 0.5 + 0.5)
                        fr_output_log = F.to_pil_image(fr_output[0]* 0.5 + 0.5)
                        reconstructed_flare_image_log = F.to_pil_image(reconstructed_flare_image[0]* 0.5 + 0.5)
                        data.append([batch_idx + 1 , wandb.Image(flare_image_log) , wandb.Image(flare_mask_real_log) , wandb.Image(light_source_mask_real_log), wandb.Image(fr_output_log), wandb.Image(reconstructed_flare_image_log)])

                    
                    loop.set_postfix(info)

        
               
            print("Epoch : {} || Time elapsed {} || TotalLoss {}".format(epoch + 1 ,time.time()- start_time , float(running_loss/ counter)))
            if self.opt.use_wandb: 
                wandb.log(epoch + 1)
                columns = ["BatchIndex" , "FlareImage",  "FlarePredictedMask" , "LightSourceMask" , "FlareRemovalOutput" , "ReconstructedImage"]
                table = wandb.Table(data=data , columns = columns)
                wandb.log({f"{self.opt.train_mode} => {epoch + 1} predictions" : table})
            if (epoch + 1 )% 100 == 0:
                if self.opt.use_schedular:
                    utils.adjust_learning_rate(self.opt_disc_flare_removal,epoch + 1, self.initial_lr , self.opt.decay_rate)
                    utils.adjust_learning_rate(self.opt_flare ,epoch +1 ,self.initial_lr, self.opt.decay_rate)

            if (epoch + 1) % 5 == 0:
                self.save_model()




        
    def save_model(self):
        print("=> Saving Model")
        checkpoint_path = self.opt.checkpoint_dir
        torch.save(self.FR.state_dict() , os.path.join(checkpoint_path , "FR.pth"))
        torch.save(self.FG.state_dict() , os.path.join(checkpoint_path , "FG.pth"))
        torch.save(self.LSD.state_dict() , os.path.join(checkpoint_path , "LSD.pth"))
        torch.save(self.FD.state_dict() , os.path.join(checkpoint_path , "FD.pth"))
        torch.save(self.FRD.state_dict() , os.path.join(checkpoint_path , "FRD.pth"))
        torch.save(self.opt_disc_flare_removal.state_dict() , os.path.join(checkpoint_path , "opt_disc_flare_removal.pth"))
        torch.save(self.opt_flare.state_dict() , os.path.join(checkpoint_path , "opt_flare.pth"))
        print("[+] Weights saved.")


    def load_model(self):
        print("=> Loading Checkpoints")
        checkpoint_path = self.opt.checkpoint_path
        try:
            self.FR.load_state_dict(torch.load(os.path.join(checkpoint_path , "FR.pth") ,map_location = self.device))
            self.FG.load_state_dict(torch.load(os.path.join(checkpoint_path , "FG.pth") , map_location=self.device))
            self.LSD.load_state_dict(torch.load(os.path.join(checkpoint_path , "LSD.pth") , map_location=self.device))
            self.FD.load_state_dict(torch.load(os.path.join(checkpoint_path , "FD.pth") , map_location=self.device))
            self.FRD.load_state_dict(torch.load(os.path.join(checkpoint_path , "FRD.pth") , map_location=self.device))
            if self.opt.load_optimizer:
                self.opt_disc_flare_removal.load_state_dict(torch.load(os.path.join(checkpoint_path , "opt_disc_flare_removal.pth") , map_location=self.device))                
                self.opt_flare.load_state_dict(torch.load(os.path.join(checkpoint_path , "opt_flare.pth") , map_location=self.device))
                self.initial_lr = self.opt_flare.param_groups[0]['lr']
                
            print("[+] Weights Loaded")
        except FileNotFoundError as e:
            print(f"[!] {e} , skipping loading weights")
        



def main():
    opt = get_args()
    if opt.use_wandb:
        wandb.init(name=opt.wandbrunname ,project=opt.projectname , entity=opt.wandbusername)
    # To reproduce the training 
    utils.seed_everything(opt.seed)
    try:
        if opt.train_mode == "flare_free":
            tm = TrainFlareFree(opt)
            if opt.load_model:
                tm.load_model()       
            tm.train()
            tm.save_model()
        elif opt.train_mode == "flare":
            tm = TrainFlare(opt)
            if opt.load_model:
                tm.load_model()        
            tm.train()
            tm.save_model()
    except KeyboardInterrupt:
        print("[!] Keyboard Interrupt! Saving the model and Shutting down.")
        tm.save_model()


if __name__ == "__main__":
    main()
    