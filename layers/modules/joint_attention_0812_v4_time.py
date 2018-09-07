import torch.nn as nn 
import torch 
from torch.nn import functional as F
from torch.autograd import Variable

from se_module import SELayer 

class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride=1,padding=0):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self,x):
        return F.relu( self.bn( self.conv(x) ) )



class Local_Global_Attention_Hybrid_Light(nn.Module):
    def __init__(self,in_channel,channel_chunck,channel_reduction):
        super(Local_Global_Attention_Hybrid_Light,self).__init__()
        self.channel_reduction= channel_reduction
        self.in_channel = in_channel 
        self.channel_chunck = channel_chunck
        self.ses = nn.ModuleList([ SELayer(channel_chunck,channel_reduction) ] * ( in_channel//channel_chunck))
        self.nlayers = in_channel // channel_chunck
        self.global_attention_fc = nn.Sequential(
                nn.Linear( self.nlayers, 1 ),
                nn.ReLU(True),
                nn.Linear( 1, self.nlayers)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        print(self.ses[0] == self.ses[1])

        self.conv0s = nn.ModuleList([ConvBlock(channel_chunck,1,kernel_size=3,stride=1,padding=1)]*self.nlayers)
        self.conv1s = nn.ModuleList( [ ConvBlock(1,1,3,stride=2,padding=1)]*self.nlayers )
        self.conv2s = nn.ModuleList( [ ConvBlock(1,1,1)]*self.nlayers)



        self.global_spatial_decoder_top = nn.Sequential(
            ConvBlock(self.nlayers,self.nlayers,kernel_size=3,stride=2,padding=1) ,# top 
            ConvBlock(self.nlayers,1,1,1,0),
            ConvBlock(1,self.nlayers,1,1,0)
        )

        # upsample 

        self.global_spatial_decoder_down = nn.Sequential(
            ConvBlock(self.nlayers,self.nlayers,3,1,1)

        )

    def forward(self,x):

        layers = list( torch.split(x,self.channel_chunck,dim=1) )

        _,_,h,w = layers[0].size()
        for i in range(0,len(layers)):
            layers[i]  = self.ses[i](layers[i]) 
        scale_descripotr = [] 
        for i in range(0,len(layers)):
            scale_descripotr.append( torch.mean(layers[i],1,True) )
        
        scale_descripotr = torch.cat(scale_descripotr,dim=1)
        b, c, h, w = scale_descripotr.size()
        scale_weight = self.global_pool( scale_descripotr ).view(b,c)
        scale_weight = self.global_attention_fc( scale_weight ).view(b,c,1,1)
        weight = list(torch.split( scale_weight,1,dim=1 ))
        for i in range(0,len(layers)):
            layers[i] = layers[i] * weight[i]

        avgs = [] 
        for i in range(0,len(layers)):
            avg = layers[i].mean(1,keepdim=True)
            avgs.append(avg)
        
        spatial_descriptor = torch.cat(avgs,dim=1)
        spatial_descriptor = self.global_spatial_decoder_top(spatial_descriptor)
        spatial_descriptor = F.upsample(spatial_descriptor,(h,w),mode='bilinear')
        spatial_descriptor = F.sigmoid( self.global_spatial_decoder_down(spatial_descriptor))

        spatial_weights = torch.split(spatial_descriptor,1,dim=1)
        for i in range(0,len(layers)):
            layers[i] = layers[i] * spatial_weights[i]
        layers= torch.cat( layers,dim=1)

        return layers

if __name__=='__main__':
    a = torch.randn((2,32*3,8,8))
    hierar_attention = Local_Global_Attention_Hybrid_Light(32*3,32,16)
    a= Variable(a)
    a = hierar_attention(a)
    state_dict = hierar_attention.state_dict() 
    for name, value in state_dict.items():
        print(name)
    print(a.shape)