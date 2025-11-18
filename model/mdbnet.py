import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torchvision import transforms
import math

'''
This code for early and late fusion for RGB features.
'''

####################################### For 2D Feature Extractuion and Downsampling Network #######################################
"""
PCR
From Front to Rear: 3D Semantic Scene Completion through Planar Convolution and Attention-based Network
implemented based on:
https://github.com/waterljwant/SSC/blob/master/models/DDR.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F



class BasicDDR3d(nn.Module):
    def __init__(self, c, k=3, dilation=1, stride=1, residual=True):
        super(BasicDDR3d, self).__init__()
        d = dilation
        p = k // 2 * d
        s = stride
        
        self.conv_1x1xk = nn.Conv3d(c, c, (1, 1, k), stride=(1, 1, s), padding=(0, 0, p), bias=True, dilation=(1, 1, d))
        self.conv_1xkx1 = nn.Conv3d(c, c, (1, k, 1), stride=(1, s, 1), padding=(0, p, 0), bias=True, dilation=(1, d, 1))
        self.conv_kx1x1 = nn.Conv3d(c, c, (k, 1, 1), stride=(s, 1, 1), padding=(p, 0, 0), bias=True, dilation=(d, 1, 1))
        self.residual = residual

    def forward(self, x):
        y = self.conv_1x1xk(x)
        y = F.relu(y, inplace=True)
        y = self.conv_1xkx1(y)
        y = F.relu(y, inplace=True)
        y = self.conv_kx1x1(y)
        y = F.relu(y + x, inplace=True) if self.residual else F.relu(y, inplace=True)
        return y


class BottleneckDDR3d(nn.Module):
    def __init__(self, c_in, c, c_out, kernel=3, stride=1, dilation=1, residual=True):
        super(BottleneckDDR3d, self).__init__()
        s = stride
        k = kernel
        d = dilation
        p = k // 2 * d
        self.conv_in = nn.Conv3d(c_in, c, kernel_size=1, bias=False)
        self.conv1x3x3 = nn.Conv3d(c, c, (1, k, k), stride=s, padding=(0, p, p), bias=True, dilation=(1, d, d))
        self.conv3x3x1 = nn.Conv3d(c, c, (k, k, 1), stride=s, padding=(p, p, 0), bias=True, dilation=(d, d, 1))
        self.conv3x1x3 = nn.Conv3d(c, c, (k, 1, k), stride=s, padding=(p, 0, p), bias=True, dilation=(d, 1, d))
        self.conv_out = nn.Conv3d(c, c_out, kernel_size=1, bias=False)
        self.residual = residual

    def forward(self, x):
        y0 = self.conv_in(x)
        y0 = F.relu(y0, inplace=True)

        y1 = self.conv1x3x3(y0)
        y1 = F.relu(y1, inplace=True)

        y2 = self.conv3x3x1(y1) + y1
        y2 = F.relu(y2, inplace=True)

        y3 = self.conv3x1x3(y2) + y2 + y1
        y3 = F.relu(y3, inplace=True)

        y = self.conv_out(y3)

        y = F.relu(y + x, inplace=True) if self.residual else F.relu(y, inplace=True)
        return y


class DownsampleBlock3d(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=2, p=1):
        super(DownsampleBlock3d, self).__init__()
        self.conv = nn.Conv3d(c_in, c_out-c_in, kernel_size=k, stride=s, padding=p, bias=False)
        self.pool = nn.MaxPool3d(2, stride=2)
        
    def forward(self, x):
        y = torch.cat([self.conv(x), self.pool(x)], 1)
        y = F.relu(y, inplace=True)
        return y

################################################################################################################################
############################################### ITRM Block #####################################################################
################################################################################################################################
class BatchNormRelu(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.bn = nn.BatchNorm3d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, dilation=1):
        super().__init__()

        self.br11 = BatchNormRelu(out_c)

        self.c11 = nn.Conv3d(in_channels=out_c, out_channels=out_c, kernel_size=kernel_size,
                             padding='same', dilation=dilation)
        self.br12 = BatchNormRelu(out_c)
        self.c12 = nn.Conv3d(in_channels=out_c, out_channels=out_c, kernel_size=kernel_size,
                             padding='same', dilation=dilation)
        
        self.tanh = nn.Tanh()
    
    def forward(self, inputs):
        x = self.br11(inputs)
        x = self.c11(x)
        x = self.br12(x)
        x = self.c12(x)
        tn_inp = self.tanh(inputs)
       
        add1 = tn_inp + x

        return add1

####################################################################################################################################################
########################################################## 2D to 3D projection and 3D Feature extraction ###########################################
####################################################################################################################################################

class Feature_extractor_2D_3D(nn.Module):
    def __init__(self):
        super(Feature_extractor_2D_3D, self).__init__()
        self.pooling = nn.AvgPool3d(kernel_size=3, padding=1, stride=1)
        
        self.rgb_feature3d = nn.Sequential(
            DownsampleBlock3d(12, 16),#ch_in 
            BottleneckDDR3d(c_in=16, c=4, c_out=16, dilation=1, residual=True), 
            DownsampleBlock3d(16, 32),  #ch_out
            BottleneckDDR3d(c_in=32, c=8, c_out=32, dilation=1, residual=True),
        )
        
    def project2D_3D(self,feature2d,depth_3d,device):
        ch= 12
        b, c, h, w = feature2d.shape
        input_ = feature2d.view(b, c, -1)
        output = torch.zeros((b, c, 240*144*240)).to(device) #--> output: torch.Size([1, 12, 8294400])
        
        indexo, indexi = torch.where(depth_3d > 0) #--> indexi represent pixel index where depth value existed
        output[indexo, :, depth_3d[indexo, indexi]] = input_[indexo, :, indexi]  # depth_3d[indexo, indexi] include the voxel index value(vox_idx)
        output = output.view(b, c, 240, 144, 240).contiguous()
        
        return output

    def forward(self,feature2d,depth_3d, location, device):

        '''
        #project 2D feature to 3D space
        '''
        segres = self.project2D_3D(feature2d, depth_3d,device)
        
        # initt the 3D features
        pool = self.pooling(segres)
        zero = (segres == 0).float()
        pool = pool * zero
        segres = segres + pool
        
        if location == 'late':
          '''     
          extract 3D feature and downsamling
          '''
          seg_3dfea = self.rgb_feature3d(segres)
        
        else: # if location early
          seg_3dfea = segres
        
        return seg_3dfea

####################################################################################################################################################
########################################################## 2D Features extraction using pretrained model ###########################################
####################################################################################################################################################

def get_2Dfeatures(pretrained_model):
  num_classes = 12
  #model_name = "./pretrained_segformer_b5_ade-640-640"
  model_name = pretrained_model
  id2label = {0:'Empty', 1:'ceiling', 2:'floor', 3:'wall', 4:'window', 5:'chair', 6:'bed', 7:'sofa', 8:'table', 9:'tvs', 10:'furn', 11:'objs'}
  label2id = {v: k for k, v in id2label.items()}
  
  # load the pretrained model
  seg_model = SegformerForSemanticSegmentation.from_pretrained(model_name, id2label=id2label, label2id=label2id, return_dict=True, ignore_mismatched_sizes=True)
  image_processor = SegformerImageProcessor.from_pretrained(model_name, do_rescale=False) # the images alredy scaled within the dataloder
  
      # Freeze encoder parameters.
  for param in seg_model.segformer.encoder.parameters():
      param.requires_grad = False
  
  # Update the classifier layer in the decoder to produce the num_classes outputs
  seg_model.decode_head.classifier = nn.Conv2d(in_channels=768, out_channels=num_classes, kernel_size=(1, 1), stride=(1, 1))
  
  return seg_model,image_processor
  

######################################################################  
def get_activation(name): # to get the feature maps 'activations'
    activation = {}
    def hook(model, input, output):
        if isinstance(output, tuple):
          output = output[0]
        activation[name] = output.detach()
    return hook,activation
####################################################################################################################################################
########################################################## ResUNet + Depth +  RGB  #################################################################
####################################################################################################################################################

class BuildResUNet_RGB(nn.Module):
    
    def __init__(self, location):
        
        super(BuildResUNet_RGB, self).__init__()
        kernel_size = (3, 3, 3)
        
        # Projection and feature extraction
        self.feature2D3D = Feature_extractor_2D_3D()
        self.location = location
        
        # Trunk part
        """ Encoders  """
        if self.location =='early':  
          self.r1 = ResBlock(12, 12, kernel_size=kernel_size)
          self.mp1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
  
          self.conv2 = nn.Conv3d(12, 16, kernel_size=(3, 3, 3), padding='same')
          self.r2 = ResBlock(16, 16, kernel_size=kernel_size)
          self.mp2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        else: # if location late
          self.conv1 = nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding='same')
          self.r1 = ResBlock(8, 8, kernel_size=kernel_size)
          self.mp1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
  
          self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding='same')
          self.r2 = ResBlock(16, 16, kernel_size=kernel_size)
          self.mp2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
          
        ## U part.
        self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding='same')
        self.r3 = ResBlock(32, 32, kernel_size=(3, 3, 3))
        self.mp3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding='same')
        self.r4 = ResBlock(64, 64, kernel_size=(3, 3, 3), dilation=1)  # Dilation rate
        self.r5 = ResBlock(64, 64, kernel_size=(3, 3, 3), dilation=1)  # Dilation rate
        self.mp5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv6 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding='same')
        self.r6 = ResBlock(128, 128, kernel_size=(3, 3, 3), dilation=1)  # Dilation rate
        self.r7 = ResBlock(128, 128, kernel_size=(3, 3, 3), dilation=1)  # Dilation rate

        """ Transpose 1 """
        self.trans1 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        
        
        """ Decoders 1 , 2 """
        self.conv7 = nn.Conv3d(128, 64, (3, 3, 3), padding='same')
        self.d1 = ResBlock(64, 64, kernel_size=(3, 3, 3))

        self.d2 = ResBlock(64, 64, kernel_size=(3, 3, 3))

        """ Transpose 2 """

        self.trans2 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        """ Decoder 3 """
        self.conv8 = nn.Conv3d(64, 32, (3, 3, 3), padding='same')
        self.d3 = ResBlock(32, 32, kernel_size=(3, 3, 3))

        """ fin """
        self.f1 = nn.Conv3d(48, 16, kernel_size=1, padding='same')
        self.f2 = nn.Conv3d(16, 16, kernel_size=1, padding='same')
        self.f3 = nn.Conv3d(16, 12, kernel_size=1, padding='same')
        self.relu = nn.ReLU()
        
        
    def forward(self, rgb, inputs,feature2d, depth_3d, device):
        
        # Projection from 2D to 3D and feature extraction
        sematic_feat_3D = self.feature2D3D(feature2d, depth_3d,self.location, device)
        
        """ Encoders """
        #################################################################################
        if self.location == 'early':
          # fuse (by add) 3d sematic features to the F-TSDF 
          inputs = inputs + sematic_feat_3D
          add1 = self.r1(inputs)
          mp1 = self.mp1(add1) # 12
          conv2 = self.conv2(mp1)
          add2 = self.r2(conv2)
          mp2 = self.mp2(add2)  # 16
          #print("mp2", mp2.size())
          
          conv3 = self.conv3(mp2)
          add3 = self.r3(conv3)  # 32 
          mp3 = self.mp3(add3)
          #print("mp3", mp3.size()) 
  
          conv4 = self.conv4(mp3)
          add4 = self.r4(conv4) # 64 
          add5 = self.r5(add4)  
          mp5 = self.mp5(add5)
          #print("mp5", mp5.size()) 
  
          conv6 = self.conv6(mp5)
          add6 = self.r6(conv6)  # 128
          #print("add6", add6.size()) 
  
          add7 = self.r7(add6)
          #print("add7", add7.size()) 
  
          trans1 = self.trans1(add7)  # 64
          #print("trans1", trans1.size()) 
  
          """ Concat"""
          concat1 = torch.cat([trans1, add5], axis=1)  # 128
          #print("concat1", concat1.size()) 
  
          """ Decoders """
          ##########################################################################
          conv7 = self.conv7(concat1) # 64 
          
          d1 = self.d1(conv7)  
          #print("d1", d1.size())
  
          d2 = self.d2(d1)
          #print("d2", d2.size())
  
          trans2 = self.trans2(d2)  # 32
          #print("trans2", trans2.size())
  
          """ Concat"""
          concat2 = torch.cat([trans2, add3], axis=1)  # 64
          #print("concat2", concat2.size())
  
          conv8 = self.conv8(concat2)
          d3 = self.d3(conv8)  ## 32
          #print("d3", d3.size())
  
          concat3 = torch.cat([d3, mp2], axis=1)  # 48
          #print("concat3", concat3.size())
  
          """ output """
          f1 = self.f1(concat3)
          f1 = self.relu(f1)
          #print("f1", f1.size())
  
          f2 = self.f2(f1)
          f2 = self.relu(f2)
          #print("f2", f2.size())
  
          f3 = self.f3(f2)
          #print("f3", f3.size())
  
          output = f3

        else: # if location late
          conv1 = self.conv1(inputs)
          add1 = self.r1(conv1)
          mp1 = self.mp1(add1)  # 8
          #print("mp1", mp1.size())
          
          conv2 = self.conv2(mp1)
          add2 = self.r2(conv2)
          mp2 = self.mp2(add2)  # 16
          #print("mp2", mp2.size())

          conv3 = self.conv3(mp2)
          add3 = self.r3(conv3)  # 32 
          mp3 = self.mp3(add3)
          #print("mp3", mp3.size()) 
  
          conv4 = self.conv4(mp3)
          add4 = self.r4(conv4) # 64 
          add5 = self.r5(add4) 
          mp5 = self.mp5(add5)
          #print("mp5", mp5.size()) 
  
          conv6 = self.conv6(mp5)
          add6 = self.r6(conv6)  # 128
          #print("add6", add6.size()) 
  
          add7 = self.r7(add6)
          #print("add7", add7.size()) 
  
          trans1 = self.trans1(add7)  # 64
          #print("trans1", trans1.size()) 
  
          """ Concat"""
          concat1 = torch.cat([trans1, add5], axis=1)  # 128
          #print("concat1", concat1.size()) 
  
          """ Decoders """
          ##########################################################################
          conv7 = self.conv7(concat1) # 64  
          
          d1 = self.d1(conv7)  
          #print("d1", d1.size())
  
          d2 = self.d2(d1)
          #print("d2", d2.size())
  
          trans2 = self.trans2(d2)  # 32
          #print("trans2", trans2.size())
  
          """ Concat"""
          concat2 = torch.cat([trans2, add3], axis=1)  # 64
          #print("concat2", concat2.size())
  
          conv8 = self.conv8(concat2)
          #################################################################################
          # fuse (by add) 3d sematic features to the F-TSDF 
          conv8 = conv8 + sematic_feat_3D
          #################################################################################
          d3 = self.d3(conv8)  ## 32
          
          #print("d3", d3.size())
  
          concat3 = torch.cat([d3, mp2], axis=1)  # 48
          #print("concat3", concat3.size())
  
          """ output """
          f1 = self.f1(concat3)
          f1 = self.relu(f1)
          #print("f1", f1.size())
  
          f2 = self.f2(f1)
          f2 = self.relu(f2)
          #print("f2", f2.size())
  
          f3 = self.f3(f2)
          #print("f3", f3.size())
  
          output = f3

        return output 


def get_res_unet_rgb(location):
    return BuildResUNet_RGB(location)

####################################################################################################################################################
########################################################## ResUNet_Depth Only ######################################################################
####################################################################################################################################################
class BuildResUNet(nn.Module):
    def __init__(self):
        super(BuildResUNet, self).__init__()
        kernel_size = (3, 3, 3)
        
        # Trunk part
        """ Encoders  """
        self.conv1 = nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding='same')
        self.r1 = ResBlock(8, 8, kernel_size=kernel_size)
        self.mp1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding='same')
        self.r2 = ResBlock(16, 16, kernel_size=kernel_size)
        self.mp2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        ## Now start U part.
        self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding='same')
        self.r3 = ResBlock(32, 32, kernel_size=(3, 3, 3))
        self.mp3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding='same')
        self.r4 = ResBlock(64, 64, kernel_size=(3, 3, 3), dilation=1)  # Dilation rate
        self.r5 = ResBlock(64, 64, kernel_size=(3, 3, 3), dilation=1)  # Dilation rate
        self.mp5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv6 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding='same')
        self.r6 = ResBlock(128, 128, kernel_size=(3, 3, 3), dilation=1)  # Dilation rate
        self.r7 = ResBlock(128, 128, kernel_size=(3, 3, 3), dilation=1)  # Dilation rate

        """ Transpose 1 """
        self.trans1 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        """ Decoders 1 , 2 """
        self.conv7 = nn.Conv3d(128, 64, (3, 3, 3), padding='same')
        self.d1 = ResBlock(64, 64, kernel_size=(3, 3, 3))

        self.d2 = ResBlock(64, 64, kernel_size=(3, 3, 3))

        """ Transpose 2 """

        self.trans2 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        """ Decoder 3 """
        self.conv8 = nn.Conv3d(64, 32, (3, 3, 3), padding='same')
        self.d3 = ResBlock(32, 32, kernel_size=(3, 3, 3))

        """ fin """
        self.f1 = nn.Conv3d(48, 16, kernel_size=1, padding='same')
        self.f2 = nn.Conv3d(16, 16, kernel_size=1, padding='same')
        self.f3 = nn.Conv3d(16, 12, kernel_size=1, padding='same')
        self.relu = nn.ReLU()
        

    def forward(self, inputs):
    
        """ Encoders """
        conv1 = self.conv1(inputs)
        add1 = self.r1(conv1)
        mp1 = self.mp1(add1)  # 8
        #print("mp1", mp1.size())

        conv2 = self.conv2(mp1)
        add2 = self.r2(conv2)
        mp2 = self.mp2(add2)  # 16
        #print("mp2", mp2.size())

        conv3 = self.conv3(mp2)
        add3 = self.r3(conv3)  # 32
        mp3 = self.mp3(add3)
        #print("mp3", mp3.size())

        conv4 = self.conv4(mp3)
        add4 = self.r4(conv4)  # 64 
        add5 = self.r5(add4)  
        mp5 = self.mp5(add5)
        #print("mp5", mp5.size())

        conv6 = self.conv6(mp5)
        add6 = self.r6(conv6)  # 128
        #print("add6", add6.size())

        add7 = self.r7(add6)
        #print("add7", add7.size())

        trans1 = self.trans1(add7)  # 64
        #print("trans1", trans1.size())

        """ Concat"""
        concat1 = torch.cat([trans1, add5], axis=1)  # 128
        #print("concat1", concat1.size())

        """ Decoders """
        conv7 = self.conv7(concat1)
        d1 = self.d1(conv7)  # 64
        #print("d1", d1.size())

        d2 = self.d2(d1)
        #print("d2", d2.size())

        trans2 = self.trans2(d2)  # 64
        #print("trans2", trans2.size())

        """ Concat"""
        concat2 = torch.cat([trans2, add3], axis=1)  # 64
        #print("concat2", concat2.size())

        conv8 = self.conv8(concat2)
        d3 = self.d3(conv8)  # 32
        #print("d3", d3.size())

        concat3 = torch.cat([d3, mp2], axis=1)  # 48
        #print("concat3", concat3.size())

        """ output """
        f1 = self.f1(concat3)
        f1 = self.relu(f1)
        #print("f1", f1.size())

        f2 = self.f2(f1)
        f2 = self.relu(f2)
        #print("f2", f2.size())

        f3 = self.f3(f2)
        #print("f3", f3.size())
        output = f3
        return output
        
        
def get_res_unet():
    return BuildResUNet()

