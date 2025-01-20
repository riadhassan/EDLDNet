import torch
import torch.nn as nn
import torch.nn.functional as F
from network_lib.pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from network_lib.decoders import EDecoder
from torch.distributions.uniform import Uniform

class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        # self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        # x = self.upsample(x)
        return x

class EDLDNet(nn.Module):
    def __init__(self, num_classes=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, ag_ks=3, activation='relu', encoder='pvt_v2_b2', pretrain=True):
        super(EDLDNet, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        if encoder == 'pvt_v2_b0':
            self.backbone = pvt_v2_b0()
            path = './pretrained_pth/pvt/pvt_v2_b0.pth'
            channels=[256, 160, 64, 32]
        elif encoder == 'pvt_v2_b1':
            self.backbone = pvt_v2_b1()
            path = './pretrained_pth/pvt/pvt_v2_b1.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b2':
            self.backbone = pvt_v2_b2()
            path = './pretrained_pth/pvt/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b3':
            self.backbone = pvt_v2_b3()
            path = './pretrained_pth/pvt/pvt_v2_b3.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b4':
            self.backbone = pvt_v2_b4()
            path = './pretrained_pth/pvt/pvt_v2_b4.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b5':
            self.backbone = pvt_v2_b5() 
            path = './pretrained_pth/pvt/pvt_v2_b5.pth'
            channels=[512, 320, 128, 64]
        else:
            print('Encoder not implemented! Continuing with default encoder pvt_v2_b2.')
            self.backbone = pvt_v2_b2()  
            path = './pretrained_pth/pvt/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
            
        if pretrain==True and 'pvt_v2' in encoder:
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

        #   decoder initialization
        self.decoder = EDecoder(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, ag_ks=ag_ks, activation=activation)
        self.decoder_noisy = EDecoder(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, ag_ks=ag_ks, activation=activation)
        self.noise = FeatureNoise()

        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)

        print(f'Model EDLDNet created with encoder: {encoder}')

    def forward(self, x, weights=None, mode='test'):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        # encoder
        x1, x2, x3, x4 = self.backbone(x)
        self.skips = [x3, x2, x1]

        # decoder
        dec_outs = self.decoder(x4, self.skips)

        # prediction heads
        I12 = self.out_head2(dec_outs[2])
        I11 = self.out_head1(dec_outs[3])

        I12 = F.interpolate(I12, scale_factor=8, mode='bilinear')
        I11 = F.interpolate(I11, scale_factor=4, mode='bilinear')

        if mode == 'test':
            return [I12, I11]

        noisy_x4 = self.noise(x4)
        noisy_dec_outs = self.decoder_noisy(noisy_x4, self.skips)

        I22 = self.out_head2(noisy_dec_outs[2])
        I21 = self.out_head1(noisy_dec_outs[3])

        I22 = F.interpolate(I22, scale_factor=8, mode='bilinear')
        I21 = F.interpolate(I21, scale_factor=4, mode='bilinear')

        return [I21, I22, I12, I11]


        
if __name__ == '__main__':
    model = EDLDNet()
    input_tensor = torch.randn(1, 3, 352, 352)

    P = model(input_tensor)
    # print(P[0].size(), P[1].size(), P[2].size(), P[3].size(), P[4].size(), P[5].size(), P[6].size(), P[7].size())

