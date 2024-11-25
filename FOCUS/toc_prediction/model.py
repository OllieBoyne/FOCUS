import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add DSINE to the path
import sys
dsine_loc = os.path.join(os.path.dirname(__file__), 'DSINE')
sys.path.append(dsine_loc)

from FOCUS.toc_prediction.DSINE.models.conv_encoder_decoder.dense_depth import DenseDepth


MIN_KAPPA = 0.00

def uncertainty_activation(out, k = 10):
    """Activation for log of variance of uncertainty
    Normalize between -k and k"""
    return F.tanh(out) * k

def normal_activation(out):
    normal, kappa = out[:, :3, :, :], out[:, [3], :, :]
    normal = F.normalize(normal, p=2, dim=1)
    kappa = F.elu(kappa) + 1.0 + MIN_KAPPA
    return torch.cat([normal, kappa], dim=1)

def toc_activation(out):
    toc, unc = out[:, 4:7, :, :], out[:, 7:10, :, :]
    unc = uncertainty_activation(unc)
    toc = F.sigmoid(toc)
    return torch.cat([toc, unc], dim=1)

def hm_activation(out):
    hm = out[:, [10], :, :]
    hm = F.sigmoid(hm)
    return hm

def activation_fn(out):
    norm = normal_activation(out)
    toc = toc_activation(out)
    hm = hm_activation(out)
    return torch.cat([norm, toc, hm], dim=1)


class FootPredictorModel(nn.Module):
    def __init__(self, image_size=(256, 256), arch='densedepth__B5__NF2048__down2__bilinear__BN', **kwargs):
        super().__init__()

        output_dim = 4 + 6 + 1  # normal + toc + mask
        output_type = 'G'

        self.params = dict(image_size=image_size, arch=arch, output_dim=output_dim, output_type=output_type)

        _, B, NF, down, learned, BN = arch.split('__')
        self.n_net = DenseDepth(num_classes=output_dim,
                                B=int(B[1:]), NF=int(NF[2:]), BN=BN == 'BN',
                                down=int(down.split('down')[1]), learned_upsampling=learned == 'learned',
                                activation_fn=activation_fn)

        # prediction head to take [B x output_dim x H x W] predicted maps and output a binary classification
        # for left/right
        pool_size = 16
        self.footedness_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((pool_size, pool_size)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(output_dim * pool_size * pool_size, 2),
            nn.Sigmoid()
        )

        # imagenet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, img, **kwargs) -> dict:
        """Run forward pass of network

        :param img: [B, 3, H, W] *un-normalized* image
        : return: dict of predictions
        """

        img = (img - self.mean) / self.std
        res = self.n_net(img, **kwargs)  # [B, output_dim, H, W]

        predictions = {
            'norm_xyz': res[:, :3, :, :],
            'norm_unc': res[:, [3], :, :],
            'TOC': res[:, 4:7, :, :],
            'TOC_unc_log_var': res[:, 7:10, :, :],
            'hm': res[:, [10], :, :],
            'footedness': self.footedness_head(res),
        }

        predictions['norm_rgb'] = (predictions['norm_xyz'] + 1.0) / 2.0

        return predictions

    def estimate_size(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        return size_all_mb

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save_model(self, out_dir='predictor', fname='model_tmp'):
        data = {'state_dict': self.state_dict(), 'params': self.params}

        os.makedirs(out_dir, exist_ok=True)
        torch.save(data, os.path.join(out_dir, fname + '.pth'))

    @classmethod
    def load(cls, file, device='cuda', opts=None, **kwargs):
        data = torch.load(file, map_location=device, weights_only=False)

        state_dict = data['state_dict']
        model = cls(**data['params']).to(device)

        model.load_state_dict(state_dict, strict=True)

        return model

    @property
    def device(self):
        return next(self.parameters()).device
