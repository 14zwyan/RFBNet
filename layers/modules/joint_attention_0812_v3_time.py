import torch.nn as nn 
import torch 
from torch.nn import functional as F
from torch.autograd import Variable

from se_module import SELayer 

import time 

class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride=1,padding=0):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self,x):
        return F.relu( self.bn( self.conv(x) ) )



class Local_Global_Attention_Hybrid(nn.Module):
    

    def __init__(self,in_channel,channel_chunck,channel_reduction):
        super(Local_Global_Attention_Hybrid,self).__init__()
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

        self.conv1s = nn.ModuleList( [ ConvBlock(1,1,3,stride=2,padding=1)]*self.nlayers )
        self.conv2s = nn.ModuleList( [ ConvBlock(1,1,1)]*self.nlayers)



        self.global_spatial_decoder = nn.Sequential(
            ConvBlock(self.nlayers,1,1,1,0),
            ConvBlock(1,self.nlayers,1,1,0)
        )
    def forward(self,x):

        time_dict = {'split':0,'se':0,'cat':0,'mean':0,'scale_se':0,'spatial':0,
                'scale_spatial':0,'multiply':0}

        start_time = time.time()
        layers = list( torch.split(x,self.channel_chunck,dim=1) )
        time_dict['split'] = time.time() - start_time

        _,_,h,w = layers[0].size()
        start_time = time.time()
        for i in range(0,len(layers)):
            layers[i]  = self.ses[i](layers[i])
        time_dict['se'] = time.time() - start_time
        #layers = list( map( lambda x,y : y(x) , layers, self.ses ) )
        # all layes has  been channel attended, then genearates scale descriptor 
        #based on average of attended features 
        scale_descripotr = [] 
        start_time = time.time()
        for i in range(0,len(layers)):
            scale_descripotr.append( torch.mean(layers[i],1,True) )
        time_dict['mean']= time.time() - start_time

        start_time = time.time()
        scale_descripotr = torch.cat(scale_descripotr,dim=1)
        time_dict['cat'] = time.time() - start_time
        #scale_descripotr = torch.cat(list( map(lambda x : torch.mean(x,1,True), layers) ),dim=1)
        #print('scale_descripotr shape:'+str(scale_descripotr.shape))
        # c is supposed to be 3
        b, c, h, w = scale_descripotr.size()
        start_time= time.time()
        scale_weight = self.global_pool( scale_descripotr ).view(b,c)
        scale_weight = self.global_attention_fc( scale_weight ).view(b,c,1,1)
        time_dict['scale_se'] = time.time() - start_time
        #print('scale weight shape'+str(scale_weight.shape))
        start_time = time.time()
        weight = list(torch.split( scale_weight,1,dim=1 ))
        time_dict['split'] = time_dict['split'] + time.time() -start_time
        #print('weight shape',weight.shape)
        #layers = list( map( lambda x,y : x*y , layers,weight ))

        start_time = time.time()
        for i in range(0,len(layers)):
            layers[i] = layers[i] * weight[i]
        time_dict['multiply'] = time.time() - start_time

        #layers = torch.cat( layers,dim=1 )
        #After channel wise attention, pixel wise attention
        start_time = time.time() 
        avgs = [] 
        for i in range(0,len(layers)):
            avg = layers[i].mean(1,keepdim=True)
            avg = self.conv1s[i](avg)
            avg = F.upsample(avg,(h,w),mode='bilinear')
            avg = self.conv2s[i](avg)
            avgs.append(avg)
        time_dict['spatial'] = time.time() - start_time
        '''
        avg = list( map( lambda x : x.mean(1,keepdim=True) , layers ) )
        avg = list( map( lambda x,y :y(x) , avg, self.conv1s))
        avg = list( map( lambda x : F.upsample(x,(h,w),mode='bilinear'),avg) )
        '''
        start_time = time.time()
        spatial_descriptor = torch.cat(avgs,dim=1)
        time_dict['cat'] = time_dict['cat'] + time.time() - start_time
        start_time = time.time()
        spatial_descriptor = F.sigmoid( self.global_spatial_decoder(spatial_descriptor) )
        time_dict['scale_spatial'] = time.time() - start_time
        start_time = time.time()
        spatial_weights = torch.split(spatial_descriptor,1,dim=1)
        time_dict['split'] = time_dict['split'] + time.time() -start_time
        start_time = time.time()
        for i in range(0,len(layers)):
            layers[i] = layers[i] * spatial_weights[i]
        time_dict['multiply'] = time_dict['multiply'] + time.time() - start_time
        start_time = time.time()
        layers= torch.cat( layers,dim=1)
        time_dict['cat'] = time_dict['cat'] + time.time() -start_time

        total_time =0 
        for name in time_dict:
            total_time = total_time + time_dict[name]

        for name in time_dict: 
            time_dict[name] = time_dict[name]*1.0/total_time*100
            print(name + ':' + '%.4f'%(time_dict[name]),end=' ')
        print()
        '''
        spatial_descriptor = torch.cat( list( map( lambda x,y : y(x) , avg ,self.conv2s )) ,dim=1) 
        spatial_descriptor = F.sigmoid( self.global_spatial_decoder( spatial_descriptor ) )
        spatial_weights =  list(torch.split( spatial_descriptor,1,dim=1 ))
        layers = torch.cat( list( map( lambda x, y :x*y, layers, spatial_weights)) , dim=1)
        '''

        return layers,time_dict


class Local_Global_Attention_Hybrid_1(nn.Module):
    

    def __init__(self,in_channel,channel_chunck,channel_reduction):
        super(Local_Global_Attention_Hybrid_1,self).__init__()
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

        time_dict = {'split':0,'se':0,'cat':0,'mean':0,'scale_se':0,'spatial':0,
                'scale_spatial':0,'multiply':0}

        start_time = time.time()
        layers = list( torch.split(x,self.channel_chunck,dim=1) )
        time_dict['split'] = time.time() - start_time

        _,_,h,w = layers[0].size()
        start_time = time.time()
        for i in range(0,len(layers)):
            layers[i]  = self.ses[i](layers[i])
        time_dict['se'] = time.time() - start_time
        #layers = list( map( lambda x,y : y(x) , layers, self.ses ) )
        # all layes has  been channel attended, then genearates scale descriptor 
        #based on average of attended features 
        scale_descripotr = [] 
        start_time = time.time()
        for i in range(0,len(layers)):
            scale_descripotr.append( torch.mean(layers[i],1,True) )
        time_dict['mean']= time.time() - start_time

        start_time = time.time()
        scale_descripotr = torch.cat(scale_descripotr,dim=1)
        time_dict['cat'] = time.time() - start_time
        #scale_descripotr = torch.cat(list( map(lambda x : torch.mean(x,1,True), layers) ),dim=1)
        #print('scale_descripotr shape:'+str(scale_descripotr.shape))
        # c is supposed to be 3
        b, c, h, w = scale_descripotr.size()
        start_time= time.time()
        scale_weight = self.global_pool( scale_descripotr ).view(b,c)
        scale_weight = self.global_attention_fc( scale_weight ).view(b,c,1,1)
        time_dict['scale_se'] = time.time() - start_time
        #print('scale weight shape'+str(scale_weight.shape))
        start_time = time.time()
        weight = list(torch.split( scale_weight,1,dim=1 ))
        time_dict['split'] = time_dict['split'] + time.time() -start_time
        #print('weight shape',weight.shape)
        #layers = list( map( lambda x,y : x*y , layers,weight ))

        start_time = time.time()
        for i in range(0,len(layers)):
            layers[i] = layers[i] * weight[i]
        time_dict['multiply'] = time.time() - start_time

        #layers = torch.cat( layers,dim=1 )
        #After channel wise attention, pixel wise attention
        start_time = time.time() 
        avgs = [] 
        for i in range(0,len(layers)):
            '''
            avg = layers[i].mean(1,keepdim=True)
            avg = self.conv1s[i](avg)
            avg = F.upsample(avg,(h,w),mode='bilinear')
            avg = self.conv2s[i](avg)
            '''
            avg = layers[i].mean(1,keepdim=True)
            avgs.append(avg)
        time_dict['spatial'] = time.time() - start_time
        '''
        avg = list( map( lambda x : x.mean(1,keepdim=True) , layers ) )
        avg = list( map( lambda x,y :y(x) , avg, self.conv1s))
        avg = list( map( lambda x : F.upsample(x,(h,w),mode='bilinear'),avg) )
        '''
        start_time = time.time()
        spatial_descriptor = torch.cat(avgs,dim=1)
        time_dict['cat'] = time_dict['cat'] + time.time() - start_time
        start_time = time.time()

        spatial_descriptor = self.global_spatial_decoder_top(spatial_descriptor)
        spatial_descriptor = F.upsample(spatial_descriptor,(h,w),mode='bilinear')
        spatial_descriptor = F.sigmoid( self.global_spatial_decoder_down(spatial_descriptor))

        time_dict['scale_spatial'] = time.time() - start_time
        start_time = time.time()
        spatial_weights = torch.split(spatial_descriptor,1,dim=1)
        time_dict['split'] = time_dict['split'] + time.time() -start_time
        start_time = time.time()
        for i in range(0,len(layers)):
            layers[i] = layers[i] * spatial_weights[i]
        time_dict['multiply'] = time_dict['multiply'] + time.time() - start_time
        start_time = time.time()
        layers= torch.cat( layers,dim=1)
        time_dict['cat'] = time_dict['cat'] + time.time() -start_time

        total_time =0 
        for name in time_dict:
            total_time = total_time + time_dict[name]

        for name in time_dict: 
            time_dict[name] = time_dict[name]*1.0/total_time*100
            print(name + ':' + '%.4f'%(time_dict[name]),end=' ')
        print()
        '''
        spatial_descriptor = torch.cat( list( map( lambda x,y : y(x) , avg ,self.conv2s )) ,dim=1) 
        spatial_descriptor = F.sigmoid( self.global_spatial_decoder( spatial_descriptor ) )
        spatial_weights =  list(torch.split( spatial_descriptor,1,dim=1 ))
        layers = torch.cat( list( map( lambda x, y :x*y, layers, spatial_weights)) , dim=1)
        '''

        return layers,time_dict

if __name__ == '__main__':
    torch.cuda.set_device(0)
    attention = Local_Global_Attention_Hybrid(512*3,512,16)
    attention1 = Local_Global_Attention_Hybrid_1(512*3,512,16)
    attention = attention.cuda()
    attention1 = attention1.cuda()
    ratio = {'split':0,'se':0,'cat':0,'mean':0,'scale_se':0,'spatial':0,
                'scale_spatial':0,'multiply':0}
    ratio1 = {'split':0,'se':0,'cat':0,'mean':0,'scale_se':0,'spatial':0,
                'scale_spatial':0,'multiply':0}
    total_ratio = list()
    total_iter=50
    for i in range(0,total_iter):
        a = torch.randn(32,512*3,38,38)
        a = Variable(a)
        a = a.cuda()

        start = time.time()
        a,time_dict = attention(a)
        raw_time = time.time() - start 
        start = time.time()
        a,time_dict1 = attention1(a)
        new_time = time.time() - start 
        total_ratio.append( new_time*100.0/raw_time )
        print('total_ratio:'+ str( new_time*100.0/raw_time ))

        for name in ratio:
            ratio[name] = ratio[name] +time_dict[name]
        for name in ratio1:
            ratio1[name] = ratio1[name] + time_dict1[name]
        #del a
    print('total:')
    for name in ratio:
        print(name + ': '+'%.4f'%(ratio[name]*1.0/total_iter) )
    for name in ratio1:
        print(name + ":" + '%.4f'%(ratio1[name]*1.0/total_iter))
    print('Total time ratio :%.4f'%(sum(total_ratio)/total_iter))
    '''
    a = torch.randn((2,32*3,8,8))
    hierar_attention = Local_Global_Attention_Hybrid(32*3,32,16)
    a= Variable(a)
    a = hierar_attention(a
        state_dict = hierar_attention.state_dict() 
    for name, value in state_dict.items():
        print(name)
    print(a.shape

    '''