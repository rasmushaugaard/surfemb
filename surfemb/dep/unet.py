# Initially from https://github.com/usuyama/pytorch-unet (MIT License)
# Architecture slightly changed (removed some expensive high-res convolutions)
# and extended to allow multiple decoders
import torch
from torch import nn
import torchvision


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class, feat_preultimate=64, n_decoders=1):
        super().__init__()

        #  shared encoder
        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        #  n_decoders
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoders = [dict(
            layer0_1x1=convrelu(64, 64, 1, 0),
            layer1_1x1=convrelu(64, 64, 1, 0),
            layer2_1x1=convrelu(128, 128, 1, 0),
            layer3_1x1=convrelu(256, 256, 1, 0),
            layer4_1x1=convrelu(512, 512, 1, 0),
            conv_up3=convrelu(256 + 512, 512, 3, 1),
            conv_up2=convrelu(128 + 512, 256, 3, 1),
            conv_up1=convrelu(64 + 256, 256, 3, 1),
            conv_up0=convrelu(64 + 256, 128, 3, 1),
            conv_original_size=convrelu(128, feat_preultimate, 3, 1),
            conv_last=nn.Conv2d(feat_preultimate, n_class, 1),
        ) for _ in range(n_decoders)]

        # register decoder modules
        for i, decoder in enumerate(self.decoders):
            for key, val in decoder.items():
                setattr(self, f'decoder{i}_{key}', val)

    def forward(self, input, decoder_idx=None):
        if decoder_idx is None:
            assert len(self.decoders) == 1
            decoder_idx = [0]
        else:
            assert len(decoder_idx) == 1 or len(decoder_idx) == len(input)

        # encoder
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layers = [layer0, layer1, layer2, layer3, layer4]

        # decoders
        out = []
        for i, dec_idx in enumerate(decoder_idx):
            decoder = self.decoders[dec_idx]
            batch_slice = slice(None) if len(decoder_idx) == 1 else slice(i, i + 1)

            x = decoder['layer4_1x1'](layer4[batch_slice])
            x = self.upsample(x)
            for layer_idx in 3, 2, 1, 0:
                layer_slice = layers[layer_idx][batch_slice]
                layer_projection = decoder[f'layer{layer_idx}_1x1'](layer_slice)
                x = torch.cat([x, layer_projection], dim=1)
                x = decoder[f'conv_up{layer_idx}'](x)
                x = self.upsample(x)

            x = decoder['conv_original_size'](x)
            out.append(decoder['conv_last'](x))

        if len(decoder_idx) == 1:
            #  out: 1 x (B, C, H, W)
            return out[0]
        else:
            #  out: B x (1, C, H, W)
            return torch.stack(out)[:, 0]
