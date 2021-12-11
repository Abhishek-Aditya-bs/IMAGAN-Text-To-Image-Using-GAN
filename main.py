import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from datasets import TextDataset
from trainer import TrainingConfig, Trainer
from model import Generator, Discriminator, RDBGenerator
from DAMSM import RNN_ENCODER

train_set = TextDataset(data_dir="/home/vishalr/Desktop/CDSAML/Datasets/CUB/data/birds",
                        split='train',
                        base_size=256,
                        transform=transforms.Compose([
                            transforms.Resize(int(256*76/64)),
                            transforms.RandomCrop(256),
                            transforms.RandomHorizontalFlip()
                        ]))

generator = RDBGenerator(32,100)
discriminator = Discriminator(32)

print(train_set.n_words)
text_encoder = RNN_ENCODER(train_set.n_words, nhidden=256)
state_dict = torch.load('/home/vishalr/Desktop/CDSAML/DF-GAN/DAMSMencoders/bird/inception/text_encoder200.pth',map_location=lambda storage, loc: storage)
text_encoder.load_state_dict(state_dict)
text_encoder.cuda()

for p in text_encoder.parameters():
    p.requires_grad = False
text_encoder.eval()

train_config = TrainingConfig(gen_ckpt_path="generator.pt",disc_ckpt_path="discriminator.pt",ckpt_dir="DF-GANv2",logdir="DF-GANv2",batch_size=32)
trainer = Trainer(generator,discriminator,text_encoder,train_set,None,train_config)
# trainer.load_model()
trainer.train()
# text_input = input()
# generator.eval()

# noise = torch.randn(1,100,device=torch.device("cuda:0"))
# hidden = text_encoder.init_hidden(1)
# word_embeddings, sentence_embeddings = text_encoder(text_input,len(text_input),hidden)
# fake_image = generator(noise,sentence_embeddings)

# save_image(fake_image, 'generated_image.png')

# for name,params in generator.named_parameters():
#     print("Generator :- isinf : ",name,torch.isinf(params))
#     print("Generator :- isNaN : ",name,torch.isnan(params))

# for name,params in discriminator.named_parameters():
#     print("Discriminator :- isinf : ",name,torch.isinf(params))
#     print("Discriminator :- isNaN : ",name,torch.isnan(params))


