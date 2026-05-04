"""Pack anomaly detection inputs into mmengine format."""

from typing import Dict

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from baoiad.registry import TRANSFORMS
from baoiad.structures import ADDataSample


@TRANSFORMS.register_module(force=True)
class PackADInputs(BaseTransform):
    """Pack image, labels, and masks into mmengine DataSample format.

    Required keys: img, gt_label
    Optional keys: gt_mask, cls_name, img_path, defect_type
    """

    def transform(self, results: Dict) -> Dict:
        # Image: HWC -> CHW, numpy -> tensor
        img = results['img']
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = img[..., np.newaxis]
            img = torch.from_numpy(img.transpose(2, 0, 1)).contiguous().float()

        data_sample = ADDataSample()

        # Meta fields: metadata about the sample
        metainfo = {
            'cls_name': results.get('cls_name', ''),
            'img_path': results.get('img_path', ''),
            'defect_type': results.get('defect_type', ''),
        }
        if 'ori_img_bgr' in results:
            metainfo['ori_img_bgr'] = results['ori_img_bgr']
        if 'source_cls' in results:
            metainfo['source_cls'] = results['source_cls']
        if 'target_cls' in results:
            metainfo['target_cls'] = results['target_cls']
        data_sample.set_metainfo(metainfo)

        # Data fields: ground truth
        data_sample.gt_label = results.get('gt_label', 0)

        if 'gt_mask' in results:
            mask = results['gt_mask']
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float()
            data_sample.gt_mask = mask

        if 'support_imgs' in results:
            support_imgs = results['support_imgs']
            if isinstance(support_imgs, np.ndarray):
                if support_imgs.ndim >= 4 and support_imgs.shape[-1] <= 4:
                    axes = list(range(support_imgs.ndim))
                    axes = axes[:-3] + [axes[-1], axes[-3], axes[-2]]
                    support_imgs = support_imgs.transpose(axes)
                support_imgs = torch.from_numpy(support_imgs).contiguous().float()
            elif isinstance(support_imgs, (list, tuple)):
                support_imgs = torch.stack([
                    item if torch.is_tensor(item) else torch.as_tensor(item)
                    for item in support_imgs
                ]).float()
            data_sample.support_imgs = support_imgs

        packed = dict(
            inputs=img,
            data_samples=data_sample,
        )
        return packed


@TRANSFORMS.register_module(force=True)
class PackDRAEMInputs(BaseTransform):
    """Pack DRAEM-specific inputs including original and augmented images.

    Required keys: img, augmented_img, anomaly_mask, gt_label
    Optional keys: cls_name, img_path, defect_type

    This transform is designed for DRAEM training where:
    - img: Original image (C, H, W) in [0, 1]
    - augmented_img: Anomaly-augmented image (C, H, W) in [0, 1]
    - anomaly_mask: Binary mask for anomaly region (H, W)
    """

    def transform(self, results: Dict) -> Dict:
        # Original image: already (C, H, W) from DRAEMDataset
        img = results['img']
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = img[..., np.newaxis]
            if img.shape[-1] <= 4:  # HWC -> CHW
                img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).contiguous().float()

        # Augmented image: already (C, H, W) from DRAEMDataset
        aug_img = results['augmented_img']
        if isinstance(aug_img, np.ndarray):
            if aug_img.ndim == 2:
                aug_img = aug_img[..., np.newaxis]
            if aug_img.shape[-1] <= 4:  # HWC -> CHW
                aug_img = aug_img.transpose(2, 0, 1)
            aug_img = torch.from_numpy(aug_img).contiguous().float()

        # Anomaly mask: (H, W)
        mask = results['anomaly_mask']
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()

        data_sample = ADDataSample()

        # Meta fields
        data_sample.set_metainfo({
            'cls_name': results.get('cls_name', ''),
            'img_path': results.get('img_path', ''),
            'defect_type': results.get('defect_type', ''),
            # Store augmented image and mask as meta for DRAEM forward()
            'augmented_img': aug_img,
            'anomaly_mask': mask,
        })

        # Ground truth label
        data_sample.gt_label = results.get('gt_label', 0)

        return dict(
            inputs=img,
            data_samples=data_sample,
        )
