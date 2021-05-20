import torch
import torch.nn as nn
import numpy as np

from torchvision import datasets
import torchvision.models as models
import torch.nn.functional as torchfunc

class convMLVAE(nn.Module):
    def __init__(self, cs_dim):
        super(convMLVAE, self).__init__()
        self.cs_dim = cs_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.linear = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.c_mu = nn.Linear(1024, self.cs_dim)
        self.c_logvar = nn.Linear(1024, self.cs_dim)
        self.s_mu = nn.Linear(1024, self.cs_dim)
        self.s_logvar = nn.Linear(1024, self.cs_dim)

        self.decoder_input = nn.Sequential(
            nn.Linear(2*self.cs_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

            nn.Linear(1024, 512*4*4),
            nn.BatchNorm1d(512*4*4),
            nn.LeakyReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        s_mu, s_logvar = self.s_mu(x), self.s_logvar(x)
        c_mu, c_logvar = self.c_mu(x), self.c_logvar(x)

        return s_mu, s_logvar, c_mu, c_logvar

    def decode(self, style_z, content_z):
        x = torch.cat((style_z, content_z), dim=1)

        x = self.decoder_input(x)
        x = x.view(x.size(0), 512, 4, 4)
        output_mu = self.deconv(x)

        return output_mu


# class convMLVAE(nn.Module):
#     def __init__(self, nc, ngf, ndf, cs_dim):
#         super(convMLVAE, self).__init__()
#
#         self.nc = nc
#         self.ngf = ngf
#         self.ndf = ndf
#         self.cs_dim = cs_dim
#
#         # encoder
#         self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
#         self.bn1 = nn.BatchNorm2d(ndf)
#
#         self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
#         self.bn2 = nn.BatchNorm2d(ndf*2)
#
#         self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
#         self.bn3 = nn.BatchNorm2d(ndf*4)
#
#         self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
#         self.bn4 = nn.BatchNorm2d(ndf*8)
#
#         # self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
#         # self.bn5 = nn.BatchNorm2d(ndf*8)
#
#         self.s_mu = nn.Linear(ndf*8*4*4, self.cs_dim)
#         self.s_logvar = nn.Linear(ndf*8*4*4, self.cs_dim)
#         self.c_mu = nn.Linear(ndf * 8 * 4 * 4, self.cs_dim)
#         self.c_logvar = nn.Linear(ndf * 8 * 4 * 4, self.cs_dim)
#
#         # decoder
#         self.d1 = nn.Linear(2*self.cs_dim, ngf*8*2*4*4)
#
#         self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
#         self.pd1 = nn.ReplicationPad2d(1)
#         self.d2 = nn.Conv2d(ngf*8*2, ngf*8, 3, 1)
#         self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)
#
#         self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
#         self.pd2 = nn.ReplicationPad2d(1)
#         self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
#         self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)
#
#         self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
#         self.pd3 = nn.ReplicationPad2d(1)
#         self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
#         self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)
#
#         self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
#         self.pd4 = nn.ReplicationPad2d(1)
#         self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
#         self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)
#
#         self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
#         self.pd5 = nn.ReplicationPad2d(1)
#         self.d6 = nn.Conv2d(ngf, nc, 3, 1)
#
#         self.leakyrelu = nn.LeakyReLU(0.2)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def encode(self, x):
#         h1 = self.leakyrelu(self.bn1(self.e1(x)))
#         h2 = self.leakyrelu(self.bn2(self.e2(h1)))
#         h3 = self.leakyrelu(self.bn3(self.e3(h2)))
#         h4 = self.leakyrelu(self.bn4(self.e4(h3)))
#         # h5 = self.leakyrelu(self.bn5(self.e5(h4)))
#         print(h4.size())
#         h4 = h4.view(-1, self.ndf*8*4*4)
#
#         return self.s_mu(h4), self.s_logvar(h4), self.c_mu(h4), self.s_logvar(h4)
#
#     def decode(self, s, c):
#         h1 = self.relu(self.d1(torch.cat((s,c), dim=1)))
#         h1 = h1.view(-1, self.ngf*8*2, 4, 4)
#         h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
#         h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
#         h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
#         h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))
#
#         return self.sigmoid(self.d6(self.pd5(self.up5(h5))))
#
#     def get_latent_var(self, x):
#         mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
#         z = self.reparametrize(mu, logvar)
#         return z
#
#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
#         z = self.reparametrize(mu, logvar)
#         res = self.decode(z)
#         return res, mu, logvar




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
    def __init__(self, input_dim, l1_dim, z_dim):
        super(linearVAE, self).__init__()
        self.input_dim = input_dim
        flat_dim = int(np.prod(input_dim))

        self.linear1 = nn.Sequential(
            nn.Linear(in_features=flat_dim, out_features=l1_dim, bias=True),
            nn.ReLU()
        )

        self.mu = nn.Linear(l1_dim, z_dim, bias=True)
        self.log_var = nn.Linear(l1_dim, z_dim, bias=True)

        self.linear2 = nn.Sequential(
            nn.Linear(z_dim, l1_dim, bias=True),
            nn.ReLU(),
            nn.Linear(l1_dim, flat_dim, bias=True),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        mu = self.mu(x)
        logvar = self.log_var(x)

        return mu, logvar

    def decode(self, z):
        x = self.linear2(z)
        x = x.view((-1,) + self.input_dim)
        return x


class linearMLVAE(nn.Module):
    def __init__(self, input_dim, l1_dim, cs_dim):
        super(linearMLVAE, self).__init__()
        self.input_dim = input_dim # a tuple of the dimension of the input
        flat_dim = np.prod(input_dim)

        self.linear1 = nn.Sequential(
            nn.Linear(in_features=flat_dim, out_features=l1_dim, bias=True),
            nn.ReLU()
        )

        # style
        self.s_mu = nn.Linear(l1_dim, cs_dim, bias=True)
        self.s_logvar = nn.Linear(l1_dim, cs_dim, bias=True)
        # content
        self.c_mu = nn.Linear(l1_dim, cs_dim, bias=True)
        self.c_logvar = nn.Linear(l1_dim, cs_dim, bias=True)

        self.linear2 = nn.Sequential(
            nn.Linear(2*cs_dim, l1_dim, bias=True),
            nn.ReLU(),
            nn.Linear(l1_dim, flat_dim, bias=True),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        # style
        s_mu = self.s_mu(x)
        s_logvar = self.s_logvar(x)
        # content
        c_mu = self.c_mu(x)
        c_logvar = self.c_logvar(x)

        return s_mu, s_logvar, c_mu, c_logvar
    
    def decode(self, s, c):
        z = torch.cat((s, c), dim=1)
        x = self.linear2(z)
        x = x.view((-1,) + self.input_dim)
        return x

class linearClassifier(nn.Module):
    def __init__(self, input_dim, l1_dim, nclasses):
        super(linearClassifier, self).__init__()
        self.input_dim = input_dim  # a tuple of the dimension of the input
        flat_dim = np.prod(input_dim)

        self.linear1 = nn.Sequential(
            nn.Linear(in_features=flat_dim, out_features=l1_dim, bias=True),
            nn.ReLU()
        )

        self.linear2 = nn.Sequential(
            nn.Linear(l1_dim, nclasses, bias=True),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        p = self.linear2(x)

        return p


class TwoPathNetwork(nn.Module):
    def __init__(self, input_size, intermediate_size = 500, embedding_size = 100, output_size = 1,
                 predefined_encoder = None):
        super(TwoPathNetwork, self).__init__()
        if predefined_encoder is None:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, intermediate_size),
                nn.ReLU(),
                nn.Linear(intermediate_size, embedding_size)
            )
        else:
            self.encoder = predefined_encoder
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size*2, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, output_size),
            # nn.Sigmoid()
        )

    def embed(self, x):
        z = self.encoder(x)
        z = z.view(x.size()[0], -1)
        return self.lin(z)

    def encode(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.encoder(x)

    def classify(self, z1, z2):
        combined_var = torch.cat((z1, z2), 1)
        return self.classifier(combined_var)

    def forward(self, inp1, inp2):
        z1, z2 = self.encode(inp1), self.encode(inp2)
        y = self.classify(z1, z2)
        return y