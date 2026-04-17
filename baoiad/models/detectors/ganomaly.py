"""GANomaly anomaly detector (ACCV 2018).

Faithful reimplementation of encoder-decoder-encoder GAN architecture.
Anomaly score = MSE between latent vectors from encoder1 and encoder2.
No pixel-level anomaly map (image-level only).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.optim import OptimWrapperDict

from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import ReconstructionADModel


# ==================== Pad to next power of 2 ====================

def _pad_nextpow2(batch):
    l_dim = 2 ** math.ceil(math.log(max(*batch.shape[-2:]), 2))
    padding_w = [math.ceil((l_dim - batch.shape[-2]) / 2), math.floor((l_dim - batch.shape[-2]) / 2)]
    padding_h = [math.ceil((l_dim - batch.shape[-1]) / 2), math.floor((l_dim - batch.shape[-1]) / 2)]
    return F.pad(batch, pad=[*padding_h, *padding_w])


# ==================== Encoder ====================

class Encoder(nn.Module):
    def __init__(self, input_size, latent_vec_size, num_input_channels, n_features,
                 extra_layers=0, add_final_conv_layer=True):
        super().__init__()
        self.input_layers = nn.Sequential()
        self.input_layers.add_module(
            f'initial-conv-{num_input_channels}-{n_features}',
            nn.Conv2d(num_input_channels, n_features, kernel_size=4, stride=2, padding=4, bias=False),
        )
        self.input_layers.add_module(f'initial-relu-{n_features}', nn.LeakyReLU(0.2, inplace=True))

        self.extra_layers = nn.Sequential()
        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f'extra-layers-{layer}-{n_features}-conv',
                nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.extra_layers.add_module(f'extra-layers-{layer}-{n_features}-batchnorm', nn.BatchNorm2d(n_features))
            self.extra_layers.add_module(f'extra-layers-{layer}-{n_features}-relu', nn.LeakyReLU(0.2, inplace=True))

        self.pyramid_features = nn.Sequential()
        pyramid_dim = min(*input_size) // 2
        while pyramid_dim > 4:
            in_feat = n_features
            out_feat = n_features * 2
            self.pyramid_features.add_module(
                f'pyramid-{in_feat}-{out_feat}-conv',
                nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1, bias=False),
            )
            self.pyramid_features.add_module(f'pyramid-{out_feat}-batchnorm', nn.BatchNorm2d(out_feat))
            self.pyramid_features.add_module(f'pyramid-{out_feat}-relu', nn.LeakyReLU(0.2, inplace=True))
            n_features = out_feat
            pyramid_dim = pyramid_dim // 2

        self.final_conv_layer = None
        if add_final_conv_layer:
            self.final_conv_layer = nn.Conv2d(
                n_features, latent_vec_size, kernel_size=4, stride=1, padding=0, bias=False,
            )

    def forward(self, x):
        x = self.input_layers(x)
        x = self.extra_layers(x)
        x = self.pyramid_features(x)
        if self.final_conv_layer is not None:
            x = self.final_conv_layer(x)
        return x


# ==================== Decoder ====================

class Decoder(nn.Module):
    def __init__(self, input_size, latent_vec_size, num_input_channels, n_features, extra_layers=0):
        super().__init__()
        self.latent_input = nn.Sequential()
        exp_factor = math.ceil(math.log(min(input_size) // 2, 2)) - 2
        n_input_features = n_features * (2 ** exp_factor)

        self.latent_input.add_module(
            f'initial-{latent_vec_size}-{n_input_features}-convt',
            nn.ConvTranspose2d(latent_vec_size, n_input_features, kernel_size=4, stride=1, padding=0, bias=False),
        )
        self.latent_input.add_module(f'initial-{n_input_features}-batchnorm', nn.BatchNorm2d(n_input_features))
        self.latent_input.add_module(f'initial-{n_input_features}-relu', nn.ReLU(inplace=True))

        self.inverse_pyramid = nn.Sequential()
        pyramid_dim = min(*input_size) // 2
        while pyramid_dim > 4:
            in_feat = n_input_features
            out_feat = n_input_features // 2
            self.inverse_pyramid.add_module(
                f'pyramid-{in_feat}-{out_feat}-convt',
                nn.ConvTranspose2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1, bias=False),
            )
            self.inverse_pyramid.add_module(f'pyramid-{out_feat}-batchnorm', nn.BatchNorm2d(out_feat))
            self.inverse_pyramid.add_module(f'pyramid-{out_feat}-relu', nn.ReLU(inplace=True))
            n_input_features = out_feat
            pyramid_dim = pyramid_dim // 2

        self.extra_layers = nn.Sequential()
        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f'extra-layers-{layer}-{n_input_features}-conv',
                nn.Conv2d(n_input_features, n_input_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.extra_layers.add_module(
                f'extra-layers-{layer}-{n_input_features}-batchnorm', nn.BatchNorm2d(n_input_features),
            )
            self.extra_layers.add_module(
                f'extra-layers-{layer}-{n_input_features}-relu', nn.LeakyReLU(0.2, inplace=True),
            )

        self.final_layers = nn.Sequential()
        self.final_layers.add_module(
            f'final-{n_input_features}-{num_input_channels}-convt',
            nn.ConvTranspose2d(n_input_features, num_input_channels, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.final_layers.add_module(f'final-{num_input_channels}-tanh', nn.Tanh())

    def forward(self, x):
        x = self.latent_input(x)
        x = self.inverse_pyramid(x)
        x = self.extra_layers(x)
        return self.final_layers(x)


# ==================== Discriminator ====================

class GanomalyDiscriminator(nn.Module):
    def __init__(self, input_size, num_input_channels, n_features, extra_layers=0):
        super().__init__()
        encoder = Encoder(input_size, 1, num_input_channels, n_features, extra_layers)
        layers = []
        for block in encoder.children():
            if isinstance(block, nn.Sequential):
                layers.extend(list(block.children()))
            else:
                if block is not None:
                    layers.append(block)
        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)
        return classifier, features


# ==================== Generator ====================

class Generator(nn.Module):
    def __init__(self, input_size, latent_vec_size, num_input_channels, n_features,
                 extra_layers=0, add_final_conv_layer=True):
        super().__init__()
        self.encoder1 = Encoder(input_size, latent_vec_size, num_input_channels, n_features,
                                extra_layers, add_final_conv_layer)
        self.decoder = Decoder(input_size, latent_vec_size, num_input_channels, n_features, extra_layers)
        self.encoder2 = Encoder(input_size, latent_vec_size, num_input_channels, n_features,
                                extra_layers, add_final_conv_layer)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_image = self.decoder(latent_i)
        latent_o = self.encoder2(gen_image)
        return gen_image, latent_i, latent_o


# ==================== Weight Init ====================

def _weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


# ==================== Loss Functions ====================

class GeneratorLoss(nn.Module):
    def __init__(self, wadv=1, wcon=50, wenc=1):
        super().__init__()
        self.loss_enc = MODELS.build(dict(type='SmoothL1Loss'))
        self.loss_adv = MODELS.build(dict(type='MSELoss'))
        self.loss_con = MODELS.build(dict(type='L1Loss'))
        self.wadv = wadv
        self.wcon = wcon
        self.wenc = wenc

    def forward(self, latent_i, latent_o, images, fake, pred_real, pred_fake):
        error_enc = self.loss_enc(latent_i, latent_o)
        error_con = self.loss_con(images, fake)
        error_adv = self.loss_adv(pred_real, pred_fake)
        return error_adv * self.wadv + error_con * self.wcon + error_enc * self.wenc


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_bce = MODELS.build(dict(type='BCELoss'))

    def forward(self, pred_real, pred_fake):
        error_real = self.loss_bce(
            pred_real, torch.ones_like(pred_real),
        )
        error_fake = self.loss_bce(
            pred_fake, torch.zeros_like(pred_fake),
        )
        return (error_fake + error_real) * 0.5


# ==================== Detector ====================

@MODELS.register_module()
class GanomalyDetector(ReconstructionADModel):
    """GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training.

    Encoder-decoder-encoder architecture. Anomaly score = MSE(latent_i, latent_o).
    Image-level only (no pixel anomaly map).

    Args:
        input_size: Input image size (H, W). Default (256, 256).
        num_input_channels: Number of input channels. Default 3.
        n_features: Base feature channels. Default 64.
        latent_vec_size: Latent vector dimension. Default 100.
        extra_layers: Extra conv layers in encoder/decoder. Default 0.
        add_final_conv_layer: Add final conv in encoder. Default True.
        wadv: Adversarial loss weight. Default 1.
        wcon: Contextual (reconstruction) loss weight. Default 50.
        wenc: Encoding loss weight. Default 1.
    """

    def __init__(
        self,
        input_size=(256, 256),
        num_input_channels=3,
        n_features=64,
        latent_vec_size=100,
        extra_layers=0,
        add_final_conv_layer=True,
        wadv=1,
        wcon=50,
        wenc=1,
        enc_loss=dict(type='MSELoss'),
        adv_loss=dict(type='MSELoss'),
        con_loss=dict(type='L1Loss'),
        disc_loss=dict(type='BCELoss'),
        data_preprocessor=None,
        init_cfg=None,
        strict=False,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.input_size = tuple(input_size)
        self.strict = bool(strict)
        self.generator = Generator(
            input_size=self.input_size,
            latent_vec_size=latent_vec_size,
            num_input_channels=num_input_channels,
            n_features=n_features,
            extra_layers=extra_layers,
            add_final_conv_layer=add_final_conv_layer,
        )
        self.discriminator = GanomalyDiscriminator(
            input_size=self.input_size,
            num_input_channels=num_input_channels,
            n_features=n_features,
            extra_layers=extra_layers,
        )

        self.loss_enc = MODELS.build(enc_loss)
        self.loss_adv = MODELS.build(adv_loss)
        self.loss_con = MODELS.build(con_loss)
        self.loss_disc = MODELS.build(disc_loss)
        self.wadv = wadv
        self.wcon = wcon
        self.wenc = wenc

    def init_weights(self):
        super().init_weights()
        self.generator.apply(_weights_init)
        self.discriminator.apply(_weights_init)

    @staticmethod
    def _prepare_inputs(inputs):
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)
        return inputs

    def _generator_loss(self, latent_i, latent_o, images, fake, feat_real, feat_fake):
        error_enc = self.loss_enc(latent_o, latent_i)
        error_con = self.loss_con(images, fake)
        error_adv = self.loss_adv(feat_real, feat_fake)
        g_loss = error_adv * self.wadv + error_con * self.wcon + error_enc * self.wenc
        return {
            'loss_g': g_loss,
            'loss_g_adv': error_adv,
            'loss_g_con': error_con,
            'loss_g_enc': error_enc,
        }

    def _discriminator_loss(self, pred_real, pred_fake):
        error_real = self.loss_disc(pred_real, torch.ones_like(pred_real))
        error_fake = self.loss_disc(pred_fake, torch.zeros_like(pred_fake))
        return (error_fake + error_real) * 0.5

    def _loss_components(self, inputs):
        padded = _pad_nextpow2(inputs)
        fake, latent_i, latent_o = self.generator(padded)
        pred_real, feat_real = self.discriminator(padded)
        pred_fake, feat_fake = self.discriminator(fake)

        losses = self._generator_loss(latent_i, latent_o, padded, fake, feat_real, feat_fake)
        pred_fake_detach, _ = self.discriminator(fake.detach())
        losses['loss_d'] = self._discriminator_loss(pred_real, pred_fake_detach)
        losses['loss'] = losses['loss_g'] + losses['loss_d']
        losses['fake'] = fake
        losses['pred_real'] = pred_real
        return losses

    def train_step(self, data, optim_wrapper):
        if not self.strict:
            return super().train_step(data, optim_wrapper)

        if (
            not isinstance(optim_wrapper, OptimWrapperDict)
            or 'generator' not in optim_wrapper
            or 'discriminator' not in optim_wrapper
        ):
            raise TypeError(
                'Strict GANomaly training requires an OptimWrapperDict with generator and discriminator optimizers.'
            )

        data = self.data_preprocessor(data, True)
        inputs = self._prepare_inputs(data['inputs'])

        generator_optim = optim_wrapper['generator']
        discriminator_optim = optim_wrapper['discriminator']

        generator_optim.zero_grad()
        discriminator_optim.zero_grad()

        losses = self._loss_components(inputs)
        scaled_g_loss = generator_optim.scale_loss(losses['loss_g'])
        generator_optim.backward(scaled_g_loss, retain_graph=True)

        if generator_optim.should_update():
            generator_optim.step()
            generator_optim.zero_grad()

        discriminator_optim.zero_grad()
        pred_fake_detach, _ = self.discriminator(losses['fake'].detach())
        loss_d = self._discriminator_loss(losses['pred_real'], pred_fake_detach)
        scaled_d_loss = discriminator_optim.scale_loss(loss_d)
        discriminator_optim.backward(scaled_d_loss)

        if discriminator_optim.should_update():
            discriminator_optim.step()
            discriminator_optim.zero_grad()
            if float(loss_d.detach().item()) < 1e-5:
                self.discriminator.apply(_weights_init)

        outputs = {
            'loss_g': losses['loss_g'].detach(),
            'loss_g_adv': losses['loss_g_adv'].detach(),
            'loss_g_con': losses['loss_g_con'].detach(),
            'loss_g_enc': losses['loss_g_enc'].detach(),
            'loss_d': loss_d.detach(),
        }
        outputs['loss'] = outputs['loss_g'] + outputs['loss_d']
        return outputs

    def forward(self, inputs, data_samples=None, mode='tensor'):
        inputs = self._prepare_inputs(inputs)
        padded = _pad_nextpow2(inputs)

        if mode == 'loss':
            losses = self._loss_components(inputs)
            return {
                'loss': losses['loss'],
                'loss_g': losses['loss_g'].detach(),
                'loss_g_adv': losses['loss_g_adv'].detach(),
                'loss_g_con': losses['loss_g_con'].detach(),
                'loss_g_enc': losses['loss_g_enc'].detach(),
                'loss_d': losses['loss_d'].detach(),
            }

        elif mode == 'predict':
            fake, latent_i, latent_o = self.generator(padded)
            # Anomaly score: MSE between latent vectors
            scores = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)

            B, C, H, W = inputs.shape
            # GANomaly is image-only; keep a zero placeholder map for APIs that
            # require pred_anomaly_map, but do not encode image scores into it.
            score_map = torch.zeros(B, 1, H, W, device=scores.device, dtype=scores.dtype)
            return build_predict_results(data_samples, scores, score_map)

        # tensor mode
        fake, latent_i, latent_o = self.generator(padded)
        return fake
