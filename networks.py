import torch
import torch.nn as nn
from collections import OrderedDict

from itertools import cycle
from torchvision import datasets
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as torchfunc

import utils
from utils import transform_config, reparameterize

# implements the concatenated relu activation function
class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        x = torch.cat((x,-x),1)
        return torchfunc.relu(x)


class convVAE(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, CNN_embed_dim=100):
        super(convVAE, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # 3x64x64
        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 64, stride=2, kernel_size=4),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        # 64x31x31
        self.convblock2 = nn.Sequential(
            nn.Conv2d(64, 128, stride=2, kernel_size=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )
        # 128x14x14
        self.convblock3 = nn.Sequential(
            nn.Conv2d(128, 256, stride=2, kernel_size=4),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )
        # 256x6x6
        self.convblock4 = nn.Sequential(
            nn.Conv2d(256, 512, stride=2, kernel_size=4),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        # 512x2x2
        self.fullyconnected1 = nn.Sequential(
            nn.Linear(512*2*2, 256),
            nn.BatchNorm1d(256),
            CReLU()
        )
        # 512
        self.fullyconnected2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            CReLU()
        )
        self.stylemulayer = nn.Linear(512, self.CNN_embed_dim)
        self.contentmulayer = nn.Linear(512, self.CNN_embed_dim)
        self.stylelogvarlayer = nn.Sequential(
            nn.Linear(512, self.CNN_embed_dim),
            nn.Tanh()
        )
        self.contentlogvarlayer = nn.Sequential(
            nn.Linear(512, self.CNN_embed_dim),
            nn.Tanh()
        )


        self.deconvblock1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2*CNN_embed_dim, out_channels=256, stride=2, kernel_size=4, output_padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )
        self.deconvblock2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, stride=2, kernel_size=4, output_padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )

        self.deconvblock3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, stride=2, kernel_size=4, output_padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()
        )

        self.deconvblock4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, stride=2, kernel_size=4, output_padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.Sigmoid()
        )

        self.deconvblockmu = nn.ConvTranspose2d(in_channels=64, out_channels=3, stride=1, kernel_size=4)

        self.deconvlogvar = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=3, stride=1, kernel_size=3),
            nn.Tanh()
        )
    
    def encode(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = x.flatten(start_dim=1)
        x = self.fullyconnected1(x)
        x = self.fullyconnected2(x)

        # The 5x multiplier is from the paper. No clue why
        style_mu, style_logvar = self.stylemulayer(x), 5*self.stylelogvarlayer(x)
        content_mu, content_logvar = self.contentmulayer(x), 5*self.contentlogvarlayer(x)

        return style_mu, style_logvar, content_mu, content_logvar

    def decode(self, style_z, content_z):
        x = torch.cat((style_z, content_z), dim=1)
        x = x.view(-1, self.CNN_embed_dim*2, 1,1)
        x = self.deconvblock1(x)
        x = self.deconvblock2(x)
        x = self.deconvblock3(x)
        x = self.deconvblock4(x)

        output_mu = self.deconvblockmu(x)
        output_logvar = 5*self.deconvlogvar(x)

        return output_mu





class DFCVAE(nn.Module):
    def __init__(self, z_dim=128, hidden_dims = None, alpha=1.0, beta=0.5):
        super(DFCVAE, self).__init__()
        self.z_dim = z_dim
        self.alpha = alpha
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

                # Build Encoder
        in_channels = 3
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_style_mu = nn.Linear(hidden_dims[-1]*4, z_dim)
        self.fc_style_var = nn.Linear(hidden_dims[-1]*4, z_dim)
        self.fc_content_mu = nn.Linear(hidden_dims[-1]*4, z_dim)
        self.fc_content_var = nn.Linear(hidden_dims[-1]*4, z_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(2*z_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

        self.feature_network = models.vgg19_bn(pretrained=True)

        # Freeze the pretrained feature network
        for param in self.feature_network.parameters():
            param.requires_grad = False

        self.feature_network.eval()
    
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        style_mu, style_logvar = self.fc_style_mu(x), self.fc_style_var(x)
        content_mu, content_logvar = self.fc_content_mu(x), self.fc_content_var(x)

        return style_mu, style_logvar, content_mu, content_logvar
    
    def decode(self, style_z, content_z):
        x = torch.cat((style_z, content_z), dim=1)
        x = self.decoder_input(x)
        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        x = self.final_layer(x)

        return x
    
    def extract_features(self, input, feature_layers = None):
        """
        Extracts the features from the pretrained model
        at the layers indicated by feature_layers.
        :param input: (Tensor) [B x C x H x W]
        :param feature_layers: List of string of IDs
        :return: List of the extracted features
        """
        if feature_layers is None:
            feature_layers = ['14', '24', '34', '43']
        features = []
        result = input
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if(key in feature_layers):
                features.append(result)

        return features





class resnetVAE(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, z_dim=128):
        super(resnetVAE, self).__init__()
        self.fc_hidden1, self.fc_hidden2, self.z_dim = fc_hidden1, fc_hidden2, z_dim

        # encoding components
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.block1 = nn.Sequential(
            nn.Linear(resnet.fc.in_features, self.fc_hidden1),
            nn.BatchNorm1d(self.fc_hidden1, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Linear(self.fc_hidden1, self.fc_hidden2),
            nn.BatchNorm1d(self.fc_hidden2, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        # latent class mu and sigma
        self.style_mu = nn.Linear(self.fc_hidden2, self.z_dim)
        self.style_logvar = nn.Linear(self.fc_hidden2, self.z_dim)
        self.content_mu = nn.Linear(self.fc_hidden2,self.z_dim)
        self.content_logvar = nn.Linear(self.fc_hidden2, self.z_dim)

        # decoding components
        self.block3 = nn.Sequential(
            nn.Linear(2*self.z_dim, self.fc_hidden2),
            nn.BatchNorm1d(self.fc_hidden2),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Linear(self.fc_hidden2, 64*4*4),
            nn.BatchNorm1d(64*4*4),
            nn.ReLU(inplace=True)
        )

        self.convTrans5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()   # restrict the range to (0,1) because input image are RBG colors in (0,1)
        )

        self.feature_network = models.vgg19_bn(pretrained=True)

        # Freeze the pretrained feature network
        for param in self.feature_network.parameters():
            param.requires_grad = False

        self.feature_network.eval()
    
    def encode(self, x):
        with torch.no_grad():
            x = self.resnet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.block2(self.block1(x))

        style_mu, style_logvar = self.style_mu(x), self.style_logvar(x)
        content_mu, content_logvar = self.content_mu(x), self.content_logvar(x)

        return style_mu, style_logvar, content_mu, content_logvar
    
    def decode(self, style_z, content_z):
        x = torch.cat((style_z, content_z), dim=1)
        x = self.block3(x)
        x = self.block4(x).view(-1, 64, 4, 4)
        x = self.convTrans7(self.convTrans6(self.convTrans5(x)))
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear')

        return x
    
    def extract_features(self, input, feature_layers = None):
        """
        Extracts the features from the pretrained model
        at the layers indicated by feature_layers.
        :param input: (Tensor) [B x C x H x W]
        :param feature_layers: List of string of IDs
        :return: List of the extracted features
        """
        if feature_layers is None:
            feature_layers = ['14', '24', '34', '43']
        features = []
        result = input
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if(key in feature_layers):
                features.append(result)

        return features





class z_classifier(nn.Module):
    def __init__(self, z_dim=128):
        super(z_classifier, self).__init__()
        self.z_dim = z_dim
        self.block1 = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Linear(32, 1)
        )
        print(True)
        
    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))
        return x





class linearVAE(nn.Module):
    def __init__(self, style_dim, class_dim):
        super(linearVAE, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_features=784, out_features=500, bias=True),
            nn.Tanh()
        )

        # style
        self.style_mu = nn.Linear(500, style_dim, bias=True)
        self.style_logvar = nn.Linear(500, style_dim, bias=True)

        # class
        self.class_mu = nn.Linear(500, class_dim, bias=True)
        self.class_logvar = nn.Linear(500, class_dim, bias=True)

        self.linear2 = nn.Sequential(
            nn.Linear(style_dim + class_dim, 500, bias=True),
            nn.Tanh(),
            nn.Linear(500, 784, bias=True),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.linear(x)

        style_z_mu = self.style_mu(x)
        style_z_logvar = self.style_logvar(x)

        class_z_mu = self.class_mu(x)
        class_z_logvar = self.class_logvar(x)

        return style_z_mu, style_z_logvar, class_z_mu, class_z_logvar
    
    def decode(self, style_z, class_z):
        x = torch.cat((style_z, class_z), dim=1)
        x = self.linear2(x)

        return x

# linear vae for 64 by 64 by 3
class linearVAE2(nn.Module):
    def __init__(self, style_dim, class_dim):
        super(linearVAE2, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_features=12288, out_features=5000, bias=True),
            nn.Tanh()
        )

        # style
        self.style_mu = nn.Linear(5000, style_dim, bias=True)
        self.style_logvar = nn.Linear(5000, style_dim, bias=True)

        # class
        self.class_mu = nn.Linear(5000, class_dim, bias=True)
        self.class_logvar = nn.Linear(5000, class_dim, bias=True)

        self.linear2 = nn.Sequential(
            nn.Linear(style_dim + class_dim, 5000, bias=True),
            nn.Tanh(),
            nn.Linear(5000, 12288, bias=True),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        style_z_mu = self.style_mu(x)
        style_z_logvar = self.style_logvar(x)

        class_z_mu = self.class_mu(x)
        class_z_logvar = self.class_logvar(x)

        return style_z_mu, style_z_logvar, class_z_mu, class_z_logvar
    
    def decode(self, style_z, class_z):
        x = torch.cat((style_z, class_z), dim=1)
        x = self.linear2(x)
        x = x.view(-1, 3, 64, 64)

        return x