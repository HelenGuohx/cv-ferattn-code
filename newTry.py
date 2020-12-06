

from torch import nn
from torch.nn import functional as F
import torch
from straightImports import DecoderBlockV2, ConvRelu, _Residual_Block_SR, STNloss, normalize_layer, Attloss

import torchvision

from torchlib.models import preactresnet




class AttentionResNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, num_filters=32, encoder_depth=34, pretrained=True):
        super(AttentionResNet, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        # attention module
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(bottom_channel_nr,                        num_filters * 8 * 2, num_filters * 8)
        self.dec5   = DecoderBlockV2(bottom_channel_nr + num_filters * 8,      num_filters * 8 * 2, num_filters * 8)
        self.dec4   = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3   = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2   = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2*2)
        self.dec1   = DecoderBlockV2(num_filters * 2 * 2,                      num_filters * 2 * 2, num_filters)

        self.attention_map = nn.Sequential(
            ConvRelu(num_filters, num_filters),
            nn.Conv2d(num_filters, out_channels, kernel_size=1)
        )


class FERAttentionNet(nn.Module):
    """FERAttentionNet
    """

    def __init__(self, dim=32, num_classes=1, num_channels=3, num_filters=32, sizeOfPool=4):

        super().__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.sizeOfPool = sizeOfPool

        # Attention module
        # TODO March 01, 2019: Include select backbone model attention
        self.attention_map = AttentionResNet(in_channels=num_channels, out_channels=num_classes, pretrained=True)

        # Feature module
        self.conv_input = nn.Conv2d(in_channels=num_channels, out_channels=num_classes, kernel_size=9, stride=1, padding=4, bias=True)
        self.feature    = self.make_layer(_Residual_Block_SR, 8, num_classes )
        self.conv_mid   = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1, padding=1, bias=True)

        # Recostruction module
        self.reconstruction = nn.Sequential(
            ConvRelu(2*num_classes+num_channels, num_filters),
            nn.Conv2d(in_channels=num_filters, out_channels=num_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.conv2_bn = nn.BatchNorm2d(num_channels)

        self.netclass = preactresnet.preactresnet18(num_classes=num_classes, num_channels=num_channels)

        # Define loss functions
        self.criterion_bce = nn.CrossEntropyLoss() # .cuda() if you have a GPU
        self.criterion_att = Attloss()
        self.criterion_stn = STNloss()




    def make_layer(self, block, num_of_layer, num_ft):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(num_ft))
        return nn.Sequential(*layers)


    def forward(self, x, x_org=None ):

        # Attention map
        g_att = self.attention_map( x )

        # Feature module
        out = self.conv_input( x )
        residual = out
        out = self.feature( out )
        out = self.conv_mid(out)
        g_ft = torch.add( out, residual )

        # Fusion
        # \sigma(A) * F(I)
        attmap = torch.mul( torch.sigmoid( g_att ) ,  g_ft )
        att = self.reconstruction( torch.cat((attmap, x, g_att), dim=1 ) )
        att = F.relu(self.conv2_bn(att))
        att_out = normalize_layer(att)

        att_pool = F.avg_pool2d(att_out, self.sizeOfPool)

        # Errors are probable here.
        y = self.netclass( att_pool )

        return y, att, g_att, g_ft

    def getLoss(self, x_org, y_mask, y_theta, y_lab_hat, y_lab, theta, att):
        attLoss = self.criterion_att(x_org, y_mask, att)
        stnLoss = self.criterion_stn(x_org, y_theta, theta)
        bceLoss = self.criterion_bce(y_lab_hat, y_lab.long())
        return attLoss + stnLoss + bceLoss



