import torch
import torch.nn as nn
import os

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base.modules import Activation


class EluPlus1(nn.Module):
    def forward(self, x):
        return nn.functional.elu(x) + 1.0


class Normalize(nn.Module):
    def forward(self, x):
        return nn.functional.normalize(x, p=2, dim=1)


class Tanh10(nn.Module):
    def forward(self, x):
        return nn.functional.tanh(x) * 10.0

_activations = {
    'elu': nn.ELU,
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'normalize': Normalize,
    'elu_plus_1': EluPlus1,
    'tanh10': Tanh10,
    None: nn.Identity,
}


class DenseHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = _activations[activation]()
        super().__init__(conv2d, upsampling, activation)


class ClassificationHeadPlus(nn.Sequential):
    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None,
                 pool_size=(16, 16)):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(pool_size) if pooling == "avg" else nn.AdaptiveMaxPool2d(pool_size)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels * pool_size[0] * pool_size[1], classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)


class FootPredictorModel(nn.Module):
    def __init__(self, encoder_name='mobilenet_v2', image_size=(256, 256), **params):
        super().__init__()

        self.params = dict(encoder_name=encoder_name, image_size=image_size)

        # ENCODER
        self.encoder = get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights="imagenet",
        )

        decoder_channels = (256, 128, 64, 32, 32)

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,  # encoder depth
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        # HEADS:
        # 1. Footedness (binary)
        # 2. Normals (H x W x 3)
        # 3. Normals uncertainty (H x W x 1)
        # 4. TOC (H x W x 3)
        # 5. TOC uncertainty (H x W x 3)
        # 6. Heatmap (H x W x 1)

        decoder_output_channels = decoder_channels[-1]
        self.heads = dict(
            footedness=ClassificationHeadPlus(decoder_output_channels, 2, activation='sigmoid'),
            norm_xyz=DenseHead(decoder_output_channels, 3, activation='normalize'),
            norm_unc=DenseHead(decoder_output_channels, 1, activation='elu_plus_1'),
            TOC=DenseHead(decoder_output_channels, 3, activation='sigmoid'),
            TOC_unc_log_var=DenseHead(decoder_output_channels, 3, activation='tanh10'),
            hm=DenseHead(decoder_output_channels, 1, activation='sigmoid'),
        )

        for k, v in self.heads.items():
            self.__setattr__(k, v)

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer('std', torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer('mean', torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.register_buffer('img_size', torch.tensor(image_size))

    def forward(self, rgb):
        """Receives image as shape [B, 3, H, W],
        returns dictionary of head predictions"""

        rgb = (rgb - self.mean) / self.std

        feat = self.encoder(rgb)
        decoder_output = self.decoder(*feat)

        predictions = dict()
        for k, head in self.heads.items():
            predictions[k] = head(decoder_output)

        # Postprocess predictions
        W, H = self.params['image_size']
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
        data = {'state_dict': self.state_dict()}
        data['params'] = self.params

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
