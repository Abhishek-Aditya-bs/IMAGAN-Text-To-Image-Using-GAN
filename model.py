import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from collections import OrderedDict

def conv3x3(in_planes, out_planes, stride=1) -> nn.Conv2d:
    """
    Returns a Conv2d object with 
    in_channels=in_planes
    out_channels=out_planes
    stride=stride
    kernel_size=(3,3)
    padding=1
    bias = False #due to batch norm layers
    """
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class upSample(nn.Module):
    def __init__(self, in_planes, out_planes, scale_factor=2):
        super(upSample,self).__init__()

        # self.conv1 = nn.ConvTranspose2d(in_channels=in_planes,out_channels=out_planes,kernel_size=4,stride=2,padding=1,bias=False)
        self.upsample = nn.Upsample(scale_factor=scale_factor,mode="nearest")
        self.conv1 = conv3x3(in_planes,out_planes)
        self.conv3x3_2 = conv3x3(in_planes=out_planes, out_planes=out_planes)
        self.conv3x3_3 = conv3x3(in_planes=out_planes, out_planes=out_planes)
        self.shortcut = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_planes,0.8)
        self.gamma  = nn.Parameter(torch.randn(1))

    def forward(self,x):

        upsample = self.upsample(x)
        out = self.conv1(upsample)
        out = F.leaky_relu(out,0.2,inplace=True)

        out = self.conv3x3_2(out)
        out = F.leaky_relu(out,0.2,inplace=True)

        out = self.conv3x3_3(out)
        out = F.leaky_relu(out,0.2,inplace=True)

        out = out + self.shortcut(upsample) * self.gamma
        out = self.batch_norm(out)
        out = F.leaky_relu(out,0.2,True)
        return out

class make_dense(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3):
        super(make_dense,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2,bias=False)
        self.bn = nn.BatchNorm2d(out_channels+in_channels,0.8)
    def forward(self,x):
        out = self.conv(x)
        out = torch.cat([out,x],dim=1)
        out = self.bn(out)
        out = F.leaky_relu(out,0.2)
        return out

class RDB(nn.Module):
    def __init__(self,in_channels,out_channels,blocks,growth_rate):
        super(RDB,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = blocks
        self.growth_rate = growth_rate

        modules = []
        for i in range(self.blocks):
            modules.append(make_dense(self.in_channels + growth_rate * i,growth_rate))

        self.layers = nn.Sequential(*modules)
        self.lff = nn.Conv2d(self.in_channels+self.growth_rate*self.blocks, self.out_channels, kernel_size=1, bias=False)
        self.shortcut = nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=3,stride=1,padding=1,bias=False)

    def forward(self,x,sentence_embeddings):
        fuse = torch.cat([x,sentence_embeddings],dim=1)
        out = self.layers(fuse)
        out = self.lff(out)
        out = F.leaky_relu(self.shortcut(fuse),negative_slope=0.2) + out

        return out

class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResBlock,self).__init__()
        self.block = nn.Sequential(
            conv3x3(in_planes=num_channels, out_planes=num_channels),
            nn.LeakyReLU(0.2,inplace=True),
            conv3x3(in_planes=num_channels, out_planes=num_channels),
        )

    def forward(self,x):
        out = self.block(x)
        out += x
        out = F.leaky_relu(out,negative_slope=0.2, inplace=True)
        
        return out


class RDBGenerator(nn.Module):
    def __init__(self, hidden_dims=32,z_dims=100):
        super(RDBGenerator,self).__init__()

        self.hidden_dims = hidden_dims
        self.z_dims = z_dims
        
        self.fc = nn.Linear(z_dims,hidden_dims * 8 * 4 * 4)
        self.upsample1 = upSample(self.hidden_dims*8,self.hidden_dims*8)
        self.upsample2 = upSample(self.hidden_dims*8,self.hidden_dims*8)
        self.encoder = RDB(in_channels=self.hidden_dims*8 + 256,out_channels=self.hidden_dims*8,blocks=8,growth_rate=64)

        self.joint = nn.Sequential(
            conv3x3(self.hidden_dims * 8 + 256, self.hidden_dims * 8), # outputs (B, 768, 16, 16)
            nn.BatchNorm2d(self.hidden_dims*8,0.8),
            nn.LeakyReLU(negative_slope=0.2,inplace=True)
        )

        self.residual_blks = self.make_layer(ResBlock, self.hidden_dims * 8)

        self.upsample3 = upSample(in_planes=self.hidden_dims * 8, out_planes=self.hidden_dims * 8) # outputs (B, 384, 32, 32)
        self.upsample4 = upSample(in_planes=self.hidden_dims * 8, out_planes=self.hidden_dims * 4) # outputs (B, 192, 64, 64)
        self.upsample5 = upSample(in_planes=self.hidden_dims * 4, out_planes=self.hidden_dims * 2) # outputs (B, 96, 128 ,128)
        self.upsample6 = upSample(in_planes=self.hidden_dims * 2, out_planes=self.hidden_dims) # outputs (B, 48, 256, 256)

        self.image = nn.Sequential(
            conv3x3(in_planes=self.hidden_dims, out_planes=3), # outputs (B, 3, 256, 256)
            nn.Tanh()
        )

    def forward(self, noise, sentence_embeddings):
        out = self.fc(noise)
        out = out.view(-1, self.hidden_dims*8, 4, 4)
        out = self.upsample1(out)
        out = self.upsample2(out)

        x = sentence_embeddings.view(-1, 256, 1,1)
        x = x.repeat(1,1,16,16)

        out = self.encoder(out,x)
        
        out = torch.cat([out,x], dim=1) #concat the image encodings and the text embeddings along the channels dimensions.
        out = self.joint(out)
        # h_code = self.joint_conv(h_code)
        out = self.residual_blks(out)

        out = self.upsample3(out)
        out = self.upsample4(out)
        out = self.upsample5(out)
        out = self.upsample6(out)

        image = self.image(out)
        return image

    def make_layer(self, block, num_channels):
        blocks = []
        for i in range(2):
            blocks.append(block(num_channels))
        return nn.Sequential(*blocks)

    def configure_optimizer(self,config):
        return optim.Adam(self.parameters(), lr=config.gen_learning_rate, betas=config.betas, eps=config.epsilon)

class affine(nn.Module):

    def __init__(self, num_features):
        super(affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(256, 256)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(256, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(256, 256)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(256, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):

        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)        

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


class G_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(G_Block,self).__init__()

        self.learnable_shortcut = in_channels != out_channels
        self.c1 = nn.Conv2d(in_channels,out_channels,3,1,1)
        self.c2 = nn.Conv2d(out_channels,out_channels,3,1,1)

        self.af0 = affine(in_channels)
        self.af1 = affine(in_channels)
        self.af2 = affine(out_channels)
        self.af3 = affine(out_channels)

        self.gamma = nn.Parameter(torch.zeros(1))

        if self.learnable_shortcut:
            self.shortcut_conv = nn.Conv2d(in_channels,out_channels,1,1,0)
        
    def forward(self, x, y=None):
        return self.shortcut(x) + self.gamma * self.residual(x,y)

    def residual(self,x,y=None):
        h = self.af0(x,y)
        h = F.leaky_relu(h,0.2,True)
        h = self.af1(h,y)
        h = F.leaky_relu(h,0.2,True)
        h = self.c1(h)

        h = self.af2(h,y)
        h = F.leaky_relu(h,0.2,True)
        h = self.af3(h,y)
        h = F.leaky_relu(h,0.2,True)
        return self.c2(h)

    def shortcut(self,x):
        if self.learnable_shortcut:
            x = self.shortcut_conv(x)
        return x

class Generator(nn.Module):
    def __init__(self,hidden_dims,z_dims):
        super(Generator,self).__init__()
        
        self.hiddem_dims = hidden_dims
        self.z_dims = z_dims

        self.fc = nn.Linear(z_dims,hidden_dims * 8 * 4 * 4)
        self.block0 = G_Block(hidden_dims * 8, hidden_dims * 8)#4x4
        self.block1 = G_Block(hidden_dims * 8, hidden_dims * 8)#4x4
        self.block2 = G_Block(hidden_dims * 8, hidden_dims * 8)#8x8
        self.block3 = G_Block(hidden_dims * 8, hidden_dims * 8)#16x16
        self.block4 = G_Block(hidden_dims * 8, hidden_dims * 4)#32x32
        self.block5 = G_Block(hidden_dims * 4, hidden_dims * 2)#64x64
        self.block6 = G_Block(hidden_dims * 2, hidden_dims * 1)#128x128
        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(self.hiddem_dims, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x, c):
        out = self.fc(x)
        out = out.view(x.size(0),8*self.hiddem_dims,4,4)
        out = self.block0(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block1(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block2(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block3(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block4(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block5(out,c)

        out = F.interpolate(out, scale_factor=2)
        out = self.block6(out,c)

        out = self.conv_img(out)

        return out

    def configure_optimizer(self,config):
        return optim.Adam(self.parameters(),lr=config.gen_learning_rate,betas=config.betas)

class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf

        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16+256, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )

    def forward(self, out, y):
        
        y = y.view(-1, 256, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)
        return out

# 定义鉴别器网络D
class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()

        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)#128
        self.block0 = resD(ndf * 1, ndf * 2)#64
        self.block1 = resD(ndf * 2, ndf * 4)#32
        self.block2 = resD(ndf * 4, ndf * 8)#16
        self.block3 = resD(ndf * 8, ndf * 16)#8
        self.block4 = resD(ndf * 16, ndf * 16)#4
        self.block5 = resD(ndf * 16, ndf * 16)#4

        self.COND_DNET = D_GET_LOGITS(ndf)

    def forward(self,x):

        out = self.conv_img(x)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        return out

    def configure_optimizer(self,config):
        return optim.Adam(self.parameters(),lr=config.disc_learning_rate, betas=config.betas)

class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = nn.Conv2d(fin,fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x)+self.gamma*self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)