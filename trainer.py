import numpy as np
import os
import time
import torch
from torch._C import memory_format
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
from datasets import prepare_data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

class TrainingConfig:

    gen_learning_rate = 0.0001
    disc_learning_rate = 0.0004
    epsilon = 1e-8
    betas=(0.00,0.9)
    max_epochs = 600
    num_workers = 6
    batch_size = 32
    drop_last=True
    shuffle = True
    pin_memory = True
    ckpt_dir = "./DF-GAN-v1/"
    gen_ckpt_path = None
    disc_ckpt_path = None
    verbose = True
    device = "cuda"
    logdir = "dfganv1"
    snap_shot = 20
    
    def __init__(self,**kwargs) -> None:
        for key,value in kwargs.items():
            setattr(self,key,value)


class Trainer:
    def __init__(self, generator, discriminator, text_encoder, train_dataset, test_dataset, configs) -> None:
        self.generator_model = generator
        self.discriminator_model = discriminator 
        self.text_encoder = text_encoder
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.configs = configs
        self.writer = SummaryWriter

        self.device = torch.device("cpu")
        
        if configs.device == "cuda" and torch.cuda.is_available():
            
            self.device = torch.cuda.current_device()
            self.generator_model = self.generator_model.to(self.device)
            self.discriminator_model = self.discriminator_model.to(self.device)
        
        elif configs.device == "cuda" and torch.cuda.is_available() == False:
            
            print("Device Error : CUDA enabled GPU not available!")
            exit(1)

        self.lossesD, self.lossesD_real, self.lossesD_wrong, self.lossesD_fake, self.losses_kl, self.losses_gen = [], [], [], [], [], []

    def save_checkpoint(self,gen_ckpt_path,disc_ckpt_path,loss_msg=None):
        gen_raw_model = self.generator_model.module if hasattr(self.generator_model, "module") else self.generator_model
        disc_raw_model = self.discriminator_model.module if hasattr(self.discriminator_model, "module") else self.discriminator_model

        ckpt_dir = self.configs.ckpt_dir

        torch.save(gen_raw_model.state_dict(),os.path.join(ckpt_dir,gen_ckpt_path))
        torch.save(disc_raw_model.state_dict(),os.path.join(ckpt_dir,disc_ckpt_path))
        if loss_msg == None:
            torch.save(torch.tensor(self.lossesD),os.path.join(ckpt_dir,"model_lossesD.pt"))
            torch.save(torch.tensor(self.lossesD_real),os.path.join(ckpt_dir,"model_lossesD_real.pt"))
            torch.save(torch.tensor(self.lossesD_fake),os.path.join(ckpt_dir,"model_lossesD_fake.pt"))
            torch.save(torch.tensor(self.losses_gen),os.path.join(ckpt_dir,"model_losses_gen.pt"))
            torch.save(torch.tensor(self.lossesD_wrong),os.path.join(ckpt_dir,"model_lossesD_wrong.pt"))
        else:
            torch.save(torch.tensor(self.lossesD),os.path.join(ckpt_dir,f"model_lossesD-snap-shot-{loss_msg}-epoch.pt"))
            torch.save(torch.tensor(self.lossesD_real),os.path.join(ckpt_dir,f"model_lossesD_real-snap-shot-{loss_msg}-epoch.pt"))
            torch.save(torch.tensor(self.lossesD_fake),os.path.join(ckpt_dir,f"model_lossesD_fake-snap-shot-{loss_msg}-epoch.pt"))
            torch.save(torch.tensor(self.losses_gen),os.path.join(ckpt_dir,f"model_losses_gen-snap-shot-{loss_msg}-epoch.pt"))
            torch.save(torch.tensor(self.lossesD_wrong),os.path.join(ckpt_dir,f"model_lossesD_wrong-snap-shot-{loss_msg}-epoch.pt"))
        print("Model Saved!")

    def load_model(self):
        ckpt_dir = self.configs.ckpt_dir
        self.load_checkpoint(self.generator_model,os.path.join(ckpt_dir, self.configs.gen_ckpt_path))
        self.load_checkpoint(self.discriminator_model, os.path.join(ckpt_dir, self.configs.disc_ckpt_path))

        # self.lossesD = torch.load(os.path.join(ckpt_dir,"model_lossesD.pt")).tolist()
        # self.lossesD_real = torch.load(os.path.join(ckpt_dir,"model_lossesD_real.pt")).tolist()
        # self.lossesD_wrong = torch.load(os.path.join(ckpt_dir,"model_lossesD_wrong.pt")).tolist()
        # self.losses_gen = torch.load(os.path.join(ckpt_dir,"model_losses_gen.pt")).tolist()
        # self.lossesD_fake = torch.load(os.path.join(ckpt_dir,"model_lossesD_fake.pt")).tolist()

    def load_checkpoint(self, model, ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))

    # def initialize_weights(self,model):
    #     weight_init(model)

    def train(self):
        generator, discriminator, text_encoder, config = self.generator_model, self.discriminator_model, self.text_encoder, self.configs

        gen_raw_model = self.generator_model.module if hasattr(self.generator_model, "module") else self.generator_model
        disc_raw_model = self.discriminator_model.module if hasattr(self.discriminator_model, "module") else self.discriminator_model

        gen_optimizer = gen_raw_model.configure_optimizer(config)
        disc_optimizer = disc_raw_model.configure_optimizer(config)

        z_dim = 100
        writer = self.writer(log_dir=f"runs/{config.logdir}/run-lr-{config.gen_learning_rate}-{time.time()}/",comment=f"lr: {config.gen_learning_rate}, batch_size:{config.batch_size}")

        #define gradient scaler for mixed precision training
        if config.device == "cuda":
            scaler = amp.GradScaler()

        def run_epoch(split):
            is_train = split == "train"
            if is_train:
                generator.train()
                discriminator.train()
            else:
                generator.eval()
                discriminator.eval()

            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(dataset=data, batch_size=config.batch_size,
                                shuffle=config.shuffle,
                                pin_memory=config.pin_memory,
                                num_workers=config.num_workers,
                                drop_last=config.drop_last)
            lossesD, lossesD_real, lossesD_wrong, lossesD_fake, losses_kl, losses_gen = [], [], [], [], [], []

            pbar = tqdm(enumerate(loader),total=len(loader)) if is_train and config.verbose else enumerate(loader)
            for it, data in pbar:
                # place data on the correct device
                print(data)
                images,captions,caption_len,class_ids, keys = prepare_data(data)
                hidden = text_encoder.init_hidden(config.batch_size)
                # print(keys)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                # with amp.autocast():    
                word_embeddings, sentence_embeddings = text_encoder(captions, caption_len, hidden)
                word_embeddings, sentence_embeddings = word_embeddings.detach(), sentence_embeddings.detach()

                images = images[0].to(self.device)
                noise = torch.randn(config.batch_size,100,device=self.device)
                
                disc_optimizer.zero_grad()
                gen_optimizer.zero_grad(set_to_none=True)
                
                with amp.autocast():
                    real_features = discriminator(images)
                    output = discriminator.COND_DNET(real_features, sentence_embeddings)
                
                errorD_real = F.relu(1.0-output).mean()

                with amp.autocast():
                    output = discriminator.COND_DNET(real_features[:(config.batch_size-1)],sentence_embeddings[1:(config.batch_size)])
                errorD_wrong = F.relu(1.0+output).mean()

                    #synthesis fake images
                with amp.autocast():
                    fake_images = generator(noise,sentence_embeddings)
                    fake_features = discriminator(fake_images.detach())
                    output = discriminator.COND_DNET(fake_features,sentence_embeddings)

                errorD_fake = F.relu(1.0+output).mean()

                errorD = errorD_real + (errorD_wrong + errorD_fake) * 0.5

                scaler.scale(errorD).backward()
                scaler.step(disc_optimizer)
                scaler.update()
                #MA-GP
                interpolated = (images.data).requires_grad_()
                sent_inter = (sentence_embeddings.data).requires_grad_()
                
                with amp.autocast():
                    features = discriminator(interpolated)
                    out = discriminator.COND_DNET(features,sent_inter)

                    grads = torch.autograd.grad(outputs=out,
                                            inputs=(interpolated,sent_inter),
                                            grad_outputs=torch.ones(out.size()).cuda(),
                                            retain_graph=True,
                                            create_graph=True,
                                            only_inputs=True)
                     
                    grad0 = grads[0].view(grads[0].size(0), -1)
                    grad1 = grads[1].view(grads[1].size(0), -1)
                    grad = torch.cat((grad0,grad1),dim=1)                        
                    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                    d_loss_gp = torch.mean((grad_l2norm) ** 6)
                    d_loss = 2.0 * d_loss_gp
                    
                disc_optimizer.zero_grad()
                gen_optimizer.zero_grad(set_to_none=True)

                scaler.scale(d_loss).backward()
                scaler.step(disc_optimizer)

    
                ### update G network ###
                disc_optimizer.zero_grad()
                gen_optimizer.zero_grad(set_to_none=True)
                with amp.autocast():
                    features = discriminator(fake_images)
                    output = discriminator.COND_DNET(features,sentence_embeddings)
                
                errorG = - output.mean()

                scaler.scale(errorG).backward()
                scaler.step(gen_optimizer)

                scaler.update()

                lossesD.append(errorD.item())
                lossesD_real.append(errorD_real.item())
                lossesD_fake.append(errorD_fake.item())
                lossesD_wrong.append(errorD_wrong.item())
                losses_gen.append(errorG.item())

                if is_train:
                    if config.verbose:
                        pbar.set_description(f"Epoch:{epoch+1} it:{it+1}| d_loss:{errorD.item()} gen_loss:{errorG.item()} real_l:{errorD_real.item()} fake_l:{errorD_fake.item()}")

            if is_train:

                self.lossesD.append(np.mean(lossesD))
                self.lossesD_real.append(np.mean(lossesD_real))
                self.lossesD_fake.append(np.mean(lossesD_fake))
                self.lossesD_wrong.append(np.mean(lossesD_wrong))

                writer.add_scalar("Loss/Loss_D",np.mean(lossesD),epoch)
                writer.add_scalar("Loss/Loss_D_real",np.mean(lossesD_real),epoch)
                writer.add_scalar("Loss/Loss_D_wrong",np.mean(lossesD_wrong),epoch)
                writer.add_scalar("Loss/Loss_Gen",np.mean(losses_gen),epoch)
                writer.add_scalar("Loss/Loss_D_fake",np.mean(lossesD_fake),epoch)
                # writer.add_scalar("Loss/Loss_KL",np.mean(losses_kl),epoch)

                with torch.no_grad():
                    image_grid_real = torchvision.utils.make_grid(
                        images[:], normalize=True
                    )
                    image_grid_fake = torchvision.utils.make_grid(
                        fake_images[:], normalize=True
                    )
                    writer.add_image("Real/Images", image_grid_real, epoch)
                    writer.add_image("Fake/Image", image_grid_fake, epoch)

            if not is_train:
                test_loss = float(np.mean(lossesD))
                if config.verbose:
                    print(f"\nEpoch:{epoch+1} | Test Loss:{test_loss}\n")
                self.test_losses.append(test_loss)
                return test_loss

        best_loss = float('inf')
        test_loss = float('inf')

        for epoch in range(config.max_epochs):
            run_epoch('train')
            
            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            good_model = self.test_dataset is None or test_loss < best_loss
            if self.configs.gen_ckpt_path is not None and self.configs.ckpt_dir is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint(self.configs.gen_ckpt_path,self.configs.disc_ckpt_path)
                if epoch % self.configs.snap_shot == 0:
                    self.save_checkpoint(self.configs.gen_ckpt_path[:-3]+"-snap-shot-"+str(epoch)+"-epoch.pt",self.configs.disc_ckpt_path[:-3]+"-snap-shot-"+str(epoch)+"-epoch.pt",epoch) 

        writer.flush()
        writer.close()