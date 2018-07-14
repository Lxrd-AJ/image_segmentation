import torch 
import torch.nn as nn 
import torch.nn.functional as F  
from torch.autograd import Variable

"""
Compound Convolution block with a ReLU activation function between the 2 blocks
"""
class UNetConv2D( nn.Module ):
    def __init__(self, in_size, out_size, batch_norm):
        super(UNetConv2D, self).__init__()
        self.kernel_size = 3
        if batch_norm:
            self.conv1 = nn.Sequential(
                            nn.Conv2d(in_size,out_size,self.kernel_size),
                            nn.BatchNorm2d(out_size),
                            nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                            nn.Conv2d(out_size, out_size,self.kernel_size),
                            nn.BatchNorm2d(out_size),
                            nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(
                            nn.Conv2d(in_size, out_size, self.kernel_size),
                            nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                            nn.Conv2d(out_size, out_size, self.kernel_size),
                            nn.ReLU()
            )
        
    def forward(self, inputs):
        c1_output = self.conv1(inputs)
        outputs = self.conv2(c1_output)
        return outputs



class UNetUpConv2D( nn.Module ):
    def __init__(self, in_size, out_size, is_deconv):
        super(UNetUpConv2D, self).__init__()
        self.conv = UNetConv2D(in_size, out_size, False)
        if is_deconv:
            self.up_conv = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up_conv = nn.UpsamplingBilinear2d(scale_factor=2)

        
    def forward(self, inputs1, inputs2):
        outputs2 = self.up_conv(inputs2)
        # Make `outputs1` equal sizes with `outputs2`
        offset = outputs2.size()[2] - inputs1.size()[2]        
        # padding = 2 * [offset // 2, offset // 2] 
        padding = 2 * [offset // 2, int(offset / 2)]
        outputs1 = F.pad(inputs1, padding)
        return self.conv( torch.cat([outputs1,outputs2],1) )



class UNet( nn.Module ):
    def __init__(self, n_classes=21, in_channels=3):
        super(UNet, self).__init__()
        self.is_deconv = True
        self.in_channels = in_channels
        self.is_batchnorm = True
        self.feature_scale = 4 

        filters = [int(x/self.feature_scale) for x in [64, 128, 256, 512, 1024]]

        # Downsampling
        self.conv1 = UNetConv2D(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UNetConv2D(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UNetConv2D(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UNetConv2D(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        
        self.center = UNetConv2D(filters[3], filters[4], self.is_batchnorm)


        # Upsampling section
        self.up_conv4 = UNetUpConv2D( filters[4], filters[3], self.is_deconv )
        self.up_conv3 = UNetUpConv2D( filters[3], filters[2], self.is_deconv )
        self.up_conv2 = UNetUpConv2D( filters[2], filters[1], self.is_deconv )
        self.up_conv1 = UNetUpConv2D( filters[1], filters[0], self.is_deconv )

        # Final layer for producing segmented outputs
        self.final = nn.Conv2d( filters[0], n_classes, 1) # 1 is the kernel size

    def forward(self, inputs):
        c1 = self.conv1(inputs)
        mp1 = self.maxpool1(c1)

        c2 = self.conv2(mp1)
        mp2 = self.maxpool2(c2)

        c3 = self.conv3(mp2)
        mp3 = self.maxpool3(c3)

        c4 = self.conv4(mp3)
        mp4 = self.maxpool4(c4)

        center = self.center(mp4)

        up4 = self.up_conv4( c4, center)
        up3 = self.up_conv3( c3, up4 )
        up2 = self.up_conv2( c2, up3 ) 
        up1 = self.up_conv1( c1, up2 )

        final = self.final(up1)

        final = F.upsample(final, inputs.size()[2:], mode='bilinear',align_corners=True) 
        final = F.sigmoid(final) 

        return final 