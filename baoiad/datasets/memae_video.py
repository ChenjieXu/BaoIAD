"""Official-style MemAE video datasets.

These datasets mirror the directory layout expected by the original MemAE
repository after MATLAB preprocessing:

    <root>/
      Train/<VideoName>/*.jpg
      Train_idx/<VideoName>/*.mat
      Test/<VideoName>/*.jpg
      Test_idx/<VideoName>/*.mat
      Test_gt/<VideoName>.mat

Each ``*_idx`` file stores one video clip as a 1-based frame index array.
The dataset returns ``C x T x H x W`` clips directly so ``MemAEDetector`` can
consume native 5D inputs without pseudo-clip adaptation.
"""

from __future__ import annotations

import os
import os.path as osp
from typing import Dict, List, Optional

import cv2
import numpy as np
import scipy.io as sio
import torch
from mmengine.dataset import BaseDataset

from baoiad.registry import DATASETS
from baoiad.structures import ADDataSample


def _sorted_frame_paths(video_dir: str) -> list[str]:
    return [
        osp.join(video_dir, name)
        for name in sorted(os.listdir(video_dir))
        if name.lower().endswith('.jpg')
    ]


def _load_frame_tensor(img_path: str, *, in_channels: int, img_size: int) -> torch.Tensor:
    if in_channels == 1:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f'Failed to load frame: {img_path}')
        if image.shape[:2] != (img_size, img_size):
            image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(image).unsqueeze(0).float()
    else:
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f'Failed to load frame: {img_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (img_size, img_size):
            image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
    return ((tensor - 127.5) / 127.5).contiguous()


@DATASETS.register_module()
class MemAEOfficialClipDataset(BaseDataset):
    """Official MemAE clip dataset backed by preprocessed video folders."""

    METAINFO: dict = dict(task='anomaly_detection')

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        dataset_name: str = 'video',
        clip_length: int = 16,
        in_channels: int = 1,
        img_size: int = 256,
        pipeline: Optional[List[dict]] = None,
        **kwargs,
    ) -> None:
        if split not in {'train', 'test'}:
            raise ValueError(f'split must be "train" or "test", got "{split}"')
        if clip_length < 1:
            raise ValueError(f'clip_length must be >= 1, got {clip_length}')
        if in_channels not in {1, 3}:
            raise ValueError(f'in_channels must be 1 or 3, got {in_channels}')

        self.split = split
        self.dataset_name = dataset_name
        self.clip_length = clip_length
        self.in_channels = in_channels
        self.img_size = img_size
        self.target_frame_offset = clip_length // 2
        self._frame_cache: dict[str, torch.Tensor] = {}
        self._label_cache: dict[str, np.ndarray] = {}

        kwargs.setdefault('serialize_data', False)
        super().__init__(data_root=data_root, pipeline=pipeline or [], **kwargs)

    @property
    def _split_dir(self) -> str:
        return 'Train' if self.split == 'train' else 'Test'

    @property
    def _idx_dir(self) -> str:
        return f'{self._split_dir}_idx'

    def _load_labels(self, video_name: str) -> np.ndarray:
        if video_name not in self._label_cache:
            gt_path = osp.join(self.data_root, 'Test_gt', f'{video_name}.mat')
            if osp.exists(gt_path):
                labels = sio.loadmat(gt_path)['l'].reshape(-1).astype(np.int64)
            else:
                labels = np.zeros(0, dtype=np.int64)
            self._label_cache[video_name] = labels
        return self._label_cache[video_name]

    def load_data_list(self) -> List[Dict]:
        idx_root = osp.join(self.data_root, self._idx_dir)
        frame_root = osp.join(self.data_root, self._split_dir)
        if not osp.isdir(idx_root) or not osp.isdir(frame_root):
            return []

        data_list: list[dict] = []
        for video_name in sorted(os.listdir(idx_root)):
            video_idx_dir = osp.join(idx_root, video_name)
            video_frame_dir = osp.join(frame_root, video_name)
            if not osp.isdir(video_idx_dir) or not osp.isdir(video_frame_dir):
                continue

            labels = self._load_labels(video_name) if self.split == 'test' else None
            for idx_file in sorted(os.listdir(video_idx_dir)):
                if not idx_file.endswith('.mat'):
                    continue
                idx_path = osp.join(video_idx_dir, idx_file)
                idx_data = sio.loadmat(idx_path)
                frame_idx = idx_data['idx'].reshape(-1).astype(np.int64)
                target_frame_number = int(frame_idx[self.target_frame_offset])
                gt_label = 0
                if labels is not None and 0 < target_frame_number <= len(labels):
                    gt_label = int(labels[target_frame_number - 1])
                data_list.append(
                    dict(
                        idx_path=idx_path,
                        video_name=video_name,
                        frame_root=video_frame_dir,
                        frame_idx=frame_idx.tolist(),
                        target_frame_number=target_frame_number,
                        gt_label=gt_label,
                        cls_name=self.dataset_name,
                        defect_type='good' if gt_label == 0 else 'anomaly',
                    )
                )
        return data_list

    def __getitem__(self, idx: int) -> Dict:
        if not self._fully_initialized:
            self.full_init()

        data_info = self.get_data_info(idx)
        frame_tensors = []
        frame_paths = []
        for frame_number in data_info['frame_idx']:
            frame_path = osp.join(data_info['frame_root'], f'{int(frame_number):03d}.jpg')
            frame_paths.append(frame_path)
            if frame_path not in self._frame_cache:
                self._frame_cache[frame_path] = _load_frame_tensor(
                    frame_path,
                    in_channels=self.in_channels,
                    img_size=self.img_size,
                )
            frame_tensors.append(self._frame_cache[frame_path].unsqueeze(1))
        clip = torch.cat(frame_tensors, dim=1)

        data_sample = ADDataSample()
        data_sample.set_metainfo(
            dict(
                cls_name=data_info['cls_name'],
                img_path=frame_paths[self.target_frame_offset],
                defect_type=data_info['defect_type'],
                video_name=data_info['video_name'],
                frame_idx=int(data_info['target_frame_number']),
            )
        )
        data_sample.gt_label = int(data_info['gt_label'])
        data_sample.gt_mask = torch.zeros((self.img_size, self.img_size), dtype=torch.float32)
        return dict(inputs=clip, data_samples=data_sample)
