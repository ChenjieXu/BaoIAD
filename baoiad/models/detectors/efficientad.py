"""EfficientAD anomaly detector.

Faithful reimplementation based on the WACV 2024 paper:
"EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies"

Aligned with anomalib implementation. Key design points:
1. PDN-Small/Medium with ImageNet normalization INSIDE forward (input must be [0,1] range)
2. AutoEncoder takes RAW IMAGES (3ch), not teacher features
3. Teacher channel mean/std computed on train set BEFORE training via build_memory_bank()
4. Quantile normalization computed on validation set BEFORE eval via compute_normalization_stats()
5. ImageNette penalty loss loaded from external dataset
6. StepLR scheduler (decay at 95% of training)
"""
import math
import os
import zipfile
import logging
import glob
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.datasets import ImageFolder
from baoiad.models.predict_utils import build_predict_results
from baoiad.registry import MODELS
from baoiad.models.base_ad_model import KnowledgeDistillationADModel

logger = logging.getLogger(__name__)

# Pretrained teacher weights from anomalib GitHub release
_WEIGHTS_URLS = [
    "https://gh-proxy.com/https://github.com/open-edge-platform/anomalib/releases/download/"
    "efficientad_pretrained_weights/efficientad_pretrained_weights.zip"
]
_WEIGHTS_URLS.append(
    "https://github.com/open-edge-platform/anomalib/releases/download/"
    "efficientad_pretrained_weights/efficientad_pretrained_weights.zip"
)
_WEIGHTS_DIR = "pre_trained/efficientad_pretrained_weights"

# ImageNette dataset for penalty loss
_IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
_IMAGENETTE_DIR = "data/imagenette2"


def _imagenet_norm(x):
    """Normalize batch with ImageNet mean/std. Input must be in [0,1] range."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _reduce_tensor_elems(tensor, max_elems=2**24):
    """Reduce tensor elements by random sampling for quantile computation."""
    tensor = torch.flatten(tensor)
    if len(tensor) > max_elems:
        perm = torch.randperm(len(tensor), device=tensor.device)
        tensor = tensor[perm[:max_elems]]
    return tensor


def _download_pretrained_weights(target_dir=_WEIGHTS_DIR):
    """Download pretrained teacher weights from anomalib release if not present."""
    small_path = os.path.join(target_dir, "pretrained_teacher_small.pth")
    medium_path = os.path.join(target_dir, "pretrained_teacher_medium.pth")
    if os.path.isfile(small_path) and os.path.isfile(medium_path):
        return

    parent_dir = os.path.dirname(target_dir) or "."
    os.makedirs(parent_dir, exist_ok=True)
    zip_path = os.path.join(parent_dir, "efficientad_pretrained_weights.zip")

    if os.path.isfile(zip_path) and not zipfile.is_zipfile(zip_path):
        logger.warning("Removing invalid EfficientAD weights archive at %s", zip_path)
        os.remove(zip_path)

    if not os.path.isfile(zip_path):
        import socket
        import urllib.request
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(30)
        last_error = None
        try:
            for url in _WEIGHTS_URLS:
                logger.info("Downloading EfficientAD pretrained teacher weights from %s...", url)
                try:
                    urllib.request.urlretrieve(url, zip_path)
                except Exception as exc:
                    last_error = exc
                    if os.path.isfile(zip_path):
                        os.remove(zip_path)
                    continue

                if zipfile.is_zipfile(zip_path):
                    last_error = None
                    break

                last_error = zipfile.BadZipFile(f"Downloaded file from {url} is not a valid zip archive.")
                if os.path.isfile(zip_path):
                    os.remove(zip_path)
        finally:
            socket.setdefaulttimeout(old_timeout)

        if last_error is not None:
            raise last_error

    if os.path.isfile(zip_path):
        logger.info("Extracting EfficientAD pretrained weights...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(parent_dir)
        os.remove(zip_path)

    os.makedirs(target_dir, exist_ok=True)
    for variant, expected_path in (
        ('small', small_path),
        ('medium', medium_path),
    ):
        if os.path.isfile(expected_path):
            continue
        candidates = glob.glob(
            os.path.join(parent_dir, '**', f'pretrained_teacher_{variant}.pth'),
            recursive=True,
        )
        if candidates:
            shutil.copy2(candidates[0], expected_path)

    if not os.path.isfile(small_path) or not os.path.isfile(medium_path):
        raise FileNotFoundError(
            f"Failed to obtain pretrained teacher weights under {target_dir}. "
            f"Please manually download from {_WEIGHTS_URLS[-1]} and extract to {target_dir}/"
        )


def _download_imagenette(target_dir=_IMAGENETTE_DIR):
    """Download ImageNette dataset if not present."""
    if os.path.isdir(os.path.join(target_dir, 'train')):
        return target_dir
    parent = os.path.dirname(target_dir) or "."
    tgz_path = os.path.join(parent, "imagenette2.tgz")
    if not os.path.isfile(tgz_path):
        import urllib.request
        logger.info("Downloading ImageNette dataset...")
        urllib.request.urlretrieve(_IMAGENETTE_URL, tgz_path)
    if os.path.isfile(tgz_path):
        import tarfile
        logger.info("Extracting ImageNette...")
        with tarfile.open(tgz_path, 'r:gz') as tf:
            tf.extractall(parent)
        os.remove(tgz_path)
    return target_dir


class PDNSmall(nn.Module):
    """PDN-Small: 4 conv + 2 avgpool. Input: [0,1] range images."""
    def __init__(self, out_channels=384, padding=False):
        super().__init__()
        pad3 = 3 if padding else 0
        pad1 = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=pad3)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=pad3)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=pad1)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 if padding else 0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 if padding else 0)

    def forward(self, x):
        x = _imagenet_norm(x)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class PDNMedium(nn.Module):
    """PDN-Medium: 6 conv + 2 avgpool. Input: [0,1] range images."""
    def __init__(self, out_channels=384, padding=False):
        super().__init__()
        pad3 = 3 if padding else 0
        pad1 = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=pad3)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=pad3)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=pad1)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=0)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 if padding else 0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 if padding else 0)

    def forward(self, x):
        x = _imagenet_norm(x)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        return x


class Encoder(nn.Module):
    """AE Encoder: 3ch input → 64ch 1x1 output (for 256x256 input)."""
    def __init__(self):
        super().__init__()
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        return self.enconv6(x)


class Decoder(nn.Module):
    """AE Decoder: 64ch → out_channels, with dropout. Matches anomalib."""
    def __init__(self, out_channels=384, padding=True):
        super().__init__()
        self.padding = padding
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)

    def forward(self, x, image_size):
        last_upsample = (
            math.ceil(image_size[0] / 4) if self.padding else math.ceil(image_size[0] / 4) - 8,
            math.ceil(image_size[1] / 4) if self.padding else math.ceil(image_size[1] / 4) - 8,
        )
        x = F.interpolate(x, size=(image_size[0] // 64 - 1, image_size[1] // 64 - 1), mode='bilinear')
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=(image_size[0] // 32, image_size[1] // 32), mode='bilinear')
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=(image_size[0] // 16 - 1, image_size[1] // 16 - 1), mode='bilinear')
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=(image_size[0] // 8, image_size[1] // 8), mode='bilinear')
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=(image_size[0] // 4 - 1, image_size[1] // 4 - 1), mode='bilinear')
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=(image_size[0] // 2 - 1, image_size[1] // 2 - 1), mode='bilinear')
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=last_upsample, mode='bilinear')
        x = F.relu(self.deconv7(x))
        return self.deconv8(x)


class AutoEncoder(nn.Module):
    """AutoEncoder: takes RAW IMAGES (3ch) with internal ImageNet normalization."""
    def __init__(self, out_channels=384, padding=False):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(out_channels, padding)

    def forward(self, x, image_size):
        x = _imagenet_norm(x)
        x = self.encoder(x)
        return self.decoder(x, image_size)


@MODELS.register_module()
class EfficientADDetector(KnowledgeDistillationADModel):
    """EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level.

    Aligned with anomalib implementation.

    IMPORTANT: Input images must be in [0,1] range (NO ImageNet normalization
    in the data pipeline). The PDN and AE apply ImageNet norm internally.

    Args:
        pdn_channels: Output channels of PDN (default 384).
        pdn_variant: 'small' or 'medium'.
        padding: Whether to use padding in PDN/AE convolutions (default False).
        pad_maps: Whether to pad anomaly maps before interpolation (default True).
        teacher_pretrained: Path to pretrained teacher PDN weights.
        imagenet_dir: Path to ImageNette dataset for penalty loss.
        data_preprocessor: Data preprocessor config.
        init_cfg: Initialization config.
    """

    def __init__(self, pdn_channels=384, pdn_variant='small', padding=False,
                 pad_maps=True, teacher_pretrained='auto',
                 ae_weight=1.0, penalty_weight=1.0,
                 imagenet_dir=_IMAGENETTE_DIR,
                 data_preprocessor=None, init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.pdn_channels = pdn_channels
        self.padding = padding
        self.pad_maps = pad_maps
        self.imagenet_dir = imagenet_dir
        self.ae_weight = ae_weight
        self.penalty_weight = penalty_weight

        PDN = PDNSmall if pdn_variant == 'small' else PDNMedium

        # Teacher (frozen)
        self.teacher = PDN(out_channels=pdn_channels, padding=padding)
        self._load_teacher_weights(teacher_pretrained, pdn_variant)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Student (trainable) — double channels for ST + AE outputs
        self.student = PDN(out_channels=2 * pdn_channels, padding=padding)

        # Autoencoder — takes raw images (3ch), NOT teacher features
        self.ae = AutoEncoder(out_channels=pdn_channels, padding=padding)

        # Teacher channel-wise mean/std (computed before training)
        self.register_buffer('teacher_mean', torch.zeros(1, pdn_channels, 1, 1))
        self.register_buffer('teacher_std', torch.zeros(1, pdn_channels, 1, 1))

        # Quantile normalization (computed before validation)
        self.register_buffer('qa_st', torch.tensor(0.0))
        self.register_buffer('qb_st', torch.tensor(0.0))
        self.register_buffer('qa_ae', torch.tensor(0.0))
        self.register_buffer('qb_ae', torch.tensor(0.0))

        # ImageNette loader for penalty loss (set up lazily)
        self._imagenet_loader = None
        self._imagenet_iter = None
        self._imagenet_setup_attempted = False

    def _is_mean_std_set(self):
        return self.teacher_mean.sum() != 0 or self.teacher_std.sum() != 0

    def _is_quantiles_set(self):
        return self.qa_st != 0 or self.qb_st != 0 or self.qa_ae != 0 or self.qb_ae != 0

    @staticmethod
    def _choose_random_aug_image(image):
        """Random augmentation: brightness, contrast, or saturation."""
        transform_functions = [
            TF.adjust_brightness, TF.adjust_contrast, TF.adjust_saturation,
        ]
        coefficient = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
        idx = int(torch.randint(0, len(transform_functions), (1,)).item())
        return transform_functions[idx](image, coefficient)

    def _load_teacher_weights(self, teacher_pretrained, pdn_variant):
        """Load pretrained teacher PDN weights."""
        if teacher_pretrained == 'auto':
            weight_file = os.path.join(
                _WEIGHTS_DIR, f"pretrained_teacher_{pdn_variant}.pth"
            )
            if not os.path.isfile(weight_file):
                try:
                    _download_pretrained_weights()
                except Exception as e:
                    import warnings
                    warnings.warn(
                        f"Failed to download pretrained teacher weights: {e}. "
                        "Teacher will be randomly initialized.",
                        UserWarning, stacklevel=3,
                    )
                    return
            logger.info(f"Loading pretrained teacher weights from {weight_file}")
            state_dict = torch.load(weight_file, map_location='cpu', weights_only=True)
            try:
                self.teacher.load_state_dict(state_dict)
            except RuntimeError as e:
                import warnings
                warnings.warn(
                    f"Skipping pretrained teacher weights due to shape mismatch: {e}. "
                    "Teacher will remain randomly initialized for this configuration.",
                    UserWarning,
                    stacklevel=3,
                )
        elif teacher_pretrained:
            state_dict = torch.load(teacher_pretrained, map_location='cpu', weights_only=True)
            self.teacher.load_state_dict(state_dict)

    def _setup_imagenet_loader(self, image_size=(256, 256)):
        """Set up ImageNette dataloader for penalty loss."""
        imagenet_dir = self.imagenet_dir
        train_dir = os.path.join(imagenet_dir, 'train')
        if not os.path.isdir(train_dir):
            try:
                _download_imagenette(imagenet_dir)
                train_dir = os.path.join(imagenet_dir, 'train')
            except Exception as e:
                logger.warning(f"Failed to download ImageNette: {e}. Penalty loss disabled.")
                return
        if not os.path.isdir(train_dir):
            logger.warning(f"ImageNette train dir not found at {train_dir}. Penalty loss disabled.")
            return
        try:
            transform = T.Compose([
                T.Resize((image_size[0] * 2, image_size[1] * 2)),
                T.RandomGrayscale(p=0.3),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ])
            dataset = ImageFolder(train_dir, transform=transform)
            self._imagenet_loader = DataLoader(
                dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=0,
            )
            self._imagenet_iter = iter(self._imagenet_loader)
            logger.info(f"ImageNette loaded: {len(dataset)} images from {train_dir}")
        except Exception as e:
            logger.warning(f"Failed to load ImageNette: {e}. Penalty loss disabled.")

    def _get_imagenet_batch(self, device):
        """Get next ImageNette batch for penalty loss."""
        if self._imagenet_loader is None:
            return None
        try:
            batch_in = next(self._imagenet_iter)[0].to(device)
        except StopIteration:
            self._imagenet_iter = iter(self._imagenet_loader)
            batch_in = next(self._imagenet_iter)[0].to(device)
        return batch_in

    def _normalize_teacher_output(self, t_feat):
        """Normalize teacher output with channel-wise mean/std."""
        if self._is_mean_std_set():
            return (t_feat - self.teacher_mean) / self.teacher_std
        return t_feat

    @torch.no_grad()
    def pre_train_setup(self, dataloader):
        """Compute teacher channel mean/std before training starts.

        Called by MemoryBankHook.before_train().
        """
        if self._is_mean_std_set():
            return  # Already computed

        device = next(self.parameters()).device
        self.eval()

        n = None
        channel_sum = None
        channel_sum_sqr = None

        for batch in dataloader:
            imgs = batch['inputs']
            if isinstance(imgs, list):
                imgs = torch.stack(imgs)
            imgs = imgs.to(device)

            y = self.teacher(imgs)
            if n is None:
                nc = y.shape[1]
                n = torch.zeros(nc, dtype=torch.int64, device=device)
                channel_sum = torch.zeros(nc, dtype=torch.float32, device=device)
                channel_sum_sqr = torch.zeros(nc, dtype=torch.float32, device=device)

            n += y[:, 0].numel()
            channel_sum += torch.sum(y, dim=[0, 2, 3])
            channel_sum_sqr += torch.sum(y ** 2, dim=[0, 2, 3])

        if n is not None:
            ch_mean = channel_sum / n
            ch_std = torch.sqrt((channel_sum_sqr / n) - (ch_mean ** 2)).float()[None, :, None, None]
            ch_mean = ch_mean.float()[None, :, None, None]
            self.teacher_mean.copy_(ch_mean)
            self.teacher_std.copy_(ch_std)
            logger.info("Teacher channel stats computed.")

        # Also set up ImageNette loader
        self._setup_imagenet_loader()
        self.train()

    @torch.no_grad()
    def compute_normalization_stats(self, dataloader, device=None):
        """Compute quantile normalization on NORMAL validation samples.

        Called before validation to calibrate anomaly maps.
        """
        if device is None:
            device = next(self.parameters()).device

        maps_st, maps_ae = [], []
        self.eval()

        for batch in dataloader:
            imgs = batch['inputs']
            if isinstance(imgs, list):
                imgs = torch.stack(imgs)
            imgs = imgs.to(device)
            data_samples = batch.get('data_samples', None)

            # Only use normal (good) images for quantile computation
            for i in range(imgs.shape[0]):
                if data_samples is not None and isinstance(data_samples, list):
                    if hasattr(data_samples[i], 'gt_label') and data_samples[i].gt_label != 0:
                        continue

                img = imgs[i:i+1]
                image_size = img.shape[-2:]
                t_out = self._normalize_teacher_output(self.teacher(img))
                s_out = self.student(img)
                ae_out = self.ae(img, image_size)

                map_st = torch.mean((t_out - s_out[:, :self.pdn_channels]) ** 2, dim=1, keepdim=True)
                map_ae = torch.mean((ae_out - s_out[:, self.pdn_channels:]) ** 2, dim=1, keepdim=True)

                if self.pad_maps:
                    map_st = F.pad(map_st, (4, 4, 4, 4))
                    map_ae = F.pad(map_ae, (4, 4, 4, 4))
                map_st = F.interpolate(map_st, size=image_size, mode='bilinear')
                map_ae = F.interpolate(map_ae, size=image_size, mode='bilinear')

                maps_st.append(map_st)
                maps_ae.append(map_ae)

        if maps_st:
            st_flat = _reduce_tensor_elems(torch.cat(maps_st))
            ae_flat = _reduce_tensor_elems(torch.cat(maps_ae))
            self.qa_st.copy_(torch.quantile(st_flat, 0.9))
            self.qb_st.copy_(torch.quantile(st_flat, 0.995))
            self.qa_ae.copy_(torch.quantile(ae_flat, 0.9))
            self.qb_ae.copy_(torch.quantile(ae_flat, 0.995))
            logger.info("Quantile normalization stats computed.")

    def forward(self, inputs, data_samples=None, mode='tensor'):
        """Forward pass.

        IMPORTANT: inputs must be in [0,1] range (no ImageNet normalization).
        """
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs)

        image_size = inputs.shape[-2:]

        with torch.no_grad():
            t_feat = self.teacher(inputs)
            t_feat = self._normalize_teacher_output(t_feat)

        s_feat = self.student(inputs)
        s_st = s_feat[:, :self.pdn_channels]
        s_ae = s_feat[:, self.pdn_channels:]
        distance_st = (t_feat - s_st) ** 2

        if mode == 'loss':
            # Setup ImageNette loader lazily on first training step (try only once)
            if self._imagenet_loader is None and not self._imagenet_setup_attempted:
                self._imagenet_setup_attempted = True
                self._setup_imagenet_loader(image_size)

            # Hard example mining for ST loss
            distance_st_flat = _reduce_tensor_elems(distance_st)
            d_hard = torch.quantile(distance_st_flat, 0.999)
            loss_hard = torch.mean(distance_st_flat[distance_st_flat >= d_hard])

            # ImageNet penalty loss
            batch_imagenet = self._get_imagenet_batch(inputs.device)
            if batch_imagenet is not None:
                student_output_penalty = self.student(batch_imagenet)[:, :self.pdn_channels]
                loss_penalty = torch.mean(student_output_penalty ** 2)
            else:
                loss_penalty = torch.tensor(0.0, device=inputs.device)

            loss_st = loss_hard + self.penalty_weight * loss_penalty

            # AE loss with augmented images
            aug_img = self._choose_random_aug_image(inputs)
            ae_out_aug = self.ae(aug_img, image_size)

            with torch.no_grad():
                t_feat_aug = self.teacher(aug_img)
                t_feat_aug = self._normalize_teacher_output(t_feat_aug)

            s_feat_aug = self.student(aug_img)
            s_ae_aug = s_feat_aug[:, self.pdn_channels:]

            loss_ae = torch.mean((t_feat_aug - ae_out_aug) ** 2)
            loss_stae = torch.mean((ae_out_aug - s_ae_aug) ** 2)

            return {'loss': loss_st + self.ae_weight * (loss_ae + loss_stae)}

        elif mode == 'predict':
            with torch.no_grad():
                ae_out = self.ae(inputs, image_size)

            map_st = torch.mean(distance_st, dim=1, keepdim=True)
            map_ae = torch.mean((ae_out - s_ae) ** 2, dim=1, keepdim=True)

            if self.pad_maps:
                map_st = F.pad(map_st, (4, 4, 4, 4))
                map_ae = F.pad(map_ae, (4, 4, 4, 4))
            map_st = F.interpolate(map_st, size=image_size, mode='bilinear', align_corners=False)
            map_ae = F.interpolate(map_ae, size=image_size, mode='bilinear', align_corners=False)

            if self._is_quantiles_set():
                map_st = 0.1 * (map_st - self.qa_st) / (self.qb_st - self.qa_st + 1e-8)
                map_ae = 0.1 * (map_ae - self.qa_ae) / (self.qb_ae - self.qa_ae + 1e-8)

            score_map = 0.5 * map_st + 0.5 * map_ae
            img_scores = torch.amax(score_map, dim=(-2, -1))

            return build_predict_results(data_samples, img_scores, score_map)

        return t_feat, s_feat

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self
