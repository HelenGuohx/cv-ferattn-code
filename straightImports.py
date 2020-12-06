from torch import nn
from torch.nn import functional as F
import torch


class Attloss(nn.Module):
    def __init__(self ):
        super(Attloss, self).__init__()
        self.maxvalueloss = 30

    def forward(self, x_org, y_mask, att):
        d = torch.exp( 6.0*torch.abs( x_org - att ) )
        loss_att = (d-1)/(d+1)
        loss_att = (loss_att).mean()
        loss_att = torch.clamp(loss_att, max=self.maxvalueloss )
        return 5.0*loss_att


class STNloss(nn.Module):

    def __init__(self):
        super(STNloss, self).__init__()

    def forward(self, x_org, y_theta, theta):
        grid_org = F.affine_grid(y_theta, x_org.size())
        grid_est = F.affine_grid(theta, x_org.size())
        x_org_t = F.grid_sample(x_org, grid_org)
        x_org_te = F.grid_sample(x_org, grid_est)
        # #loss_theta = ((y_theta - theta) ** 2).mean()
        # #loss_theta = F.mse_loss( theta.view(-1, 3*2 ) , y_theta.view(-1, 3*2 ) )
        loss_theta = ((x_org_t - x_org_te) ** 2).mean()
        return 10 * loss_theta


class AttMSEloss(nn.Module):
    def __init__(self ):
        super(AttMSEloss, self).__init__()

    def forward(self, x_org, y_mask, att):
        loss_att = ((( (x_org*y_mask[:,1,...].unsqueeze(dim=1)) - att ) ** 2)).mean()
        loss_att = ((( x_org - att ) ** 2)).mean()
        loss_att = torch.clamp(loss_att, max=30)
        return 10*loss_att


class _Residual_Block_SR(nn.Module):
    def __init__(self, num_ft):
        super(_Residual_Block_SR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu  = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output = torch.add(output, identity_data)
        return output


def normalize_layer(x):
    x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.2023 / 0.5) + (0.4914 - 0.5) / 0.5
    x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.1994 / 0.5) + (0.4822 - 0.5) / 0.5
    x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.2010 / 0.5) + (0.4465 - 0.5) / 0.5
    x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
    return x


def conv3x3(_in, _out):
    return nn.Conv2d(_in, _out, kernel_size=3, stride=1, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, _in, _out):
        super().__init__()
        self.conv = conv3x3(_in, _out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels
        self.up  = F.interpolate
        self.cr1 = ConvRelu(in_channels,     middle_channels)
        self.cr2 = ConvRelu(middle_channels, out_channels)

    def forward(self, x):
        x = self.up(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.cr2(self.cr1(x))
        return x
