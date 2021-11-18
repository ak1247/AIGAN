import torch.nn.functional as F
from torch import nn


class MNIST_target_net(nn.Module):
    def __init__(self):
        super(MNIST_target_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)

        self.fc1 = nn.Linear(64*4*4, 200)
        self.fc2 = nn.Linear(200, 200)
        self.logits = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = self.logits(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,img_channels):
        super(Discriminator, self).__init__()
        model = [
            # G_out_channels x 28 x 28
            nn.Conv2d(img_channels,8,kernel_size=4,stride=2,padding=0,bias=True),
            nn.LeakyReLU(0.2,inplace=True),
            # 8 x 13 x 13
            nn.Conv2d(8,16,kernel_size=4,stride=2,padding=0,bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 x 5 x 5
            nn.Conv2d(16,32,kernel_size=4,stride=2,padding=0,bias=False),
            #nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),      
        ]
        self.head1 = nn.Conv2d(32,1,kernel_size=1,bias=True)
        self.head2 = nn.Linear(32,10)
        
        self.model = nn.Sequential(*model)
        self.prob = nn.Sigmoid()

    def forward(self,x):
        output = self.model(x)
        out_dis = self.prob(self.head1(output).squeeze())
        out_clas = self.head2(output.view(-1,32))
        return out_dis,out_clas


class Generator(nn.Module):
    def __init__(self,img_channels):
        super(Generator, self).__init__()
        self.G_in_channels = img_channels
        self.G_out_channels = img_channels

        encoder_list = [
            # MNIST: G_in_channels*28*28
            nn.Conv2d(self.G_in_channels,8,kernel_size=3,stride=1,padding=0,bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # 8*26*26
            nn.Conv2d(8,16,kernel_size=3,stride=2,padding=0,bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # 16*12*12
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=0,bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32*5*5
        ]

        bottle_neck_list = [ResnetBlock(32),
                        ResnetBlock(32),
                        ResnetBlock(32),
                        ResnetBlock(32),
                        # 32*5*5
        ]

        decoder_list = [
            nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,padding=0,bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # state size. 16 x 11 x 11
            nn.ConvTranspose2d(16,8,kernel_size=3,stride=2,padding=0,bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # state size. 8 x 23 x 23
            nn.ConvTranspose2d(8,self.G_out_channels,kernel_size=6,stride=1,padding=0,bias=False),
            nn.Tanh()
            # state size. G_out_channels x 28 x 28
        ]

        self.encoder = nn.Sequential(*encoder_list)
        self.bottle_neck = nn.Sequential(*bottle_neck_list)
        self.decoder = nn.Sequential(*decoder_list)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self,dim,padding_type='reflect',norm_layer=nn.BatchNorm2d):
        super(ResnetBlock,self).__init__()

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim,dim,kernel_size=3,padding=p,bias=False),
                        norm_layer(dim),
                        nn.ReLU(True)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim,dim,kernel_size=3,padding=p,bias=False),
                       norm_layer(dim)]

        self.res_block =  nn.Sequential(*conv_block)



    def forward(self,x):
        x = x+self.res_block(x)
        return x
