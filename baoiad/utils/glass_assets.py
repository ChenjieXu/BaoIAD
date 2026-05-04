"""Helpers for validating strict GLASS auxiliary assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _resolve_dtd_images_root(dtd_root: Path) -> Path | None:
    if dtd_root.name == 'images' and dtd_root.is_dir():
        return dtd_root

    for candidate in (dtd_root / 'images', dtd_root / 'dtd' / 'images'):
        if candidate.is_dir():
            return candidate
    return None


def collect_glass_asset_report(
    glass_assets_root: str | Path = 'data/glass_assets/mvtec',
    dtd_root: str | Path = 'data/dtd',
) -> Dict[str, Any]:
    """Collect a structured validation report for strict GLASS assets."""
    assets_root = Path(glass_assets_root)
    fg_mask_root = assets_root / 'fg_mask'
    distribution_path = assets_root / 'mvtec_distribution.xlsx'
    dtd_root_path = Path(dtd_root)

    report: Dict[str, Any] = {
        'glass_assets_root': str(assets_root),
        'fg_mask_root': str(fg_mask_root),
        'distribution_meta_path': str(distribution_path),
        'dtd_root': str(dtd_root_path),
        'distribution_exists': distribution_path.is_file(),
        'fg_mask_root_exists': fg_mask_root.is_dir(),
    }

    required_fg_classes: list[str] = []
    distribution_rows = []
    if distribution_path.is_file():
        df = pd.read_excel(distribution_path)
        distribution_rows = df[['Class', 'Distribution', 'Foreground']].to_dict(orient='records')
        required_fg_classes = sorted(
            row['Class'].replace('mvtec_', '', 1)
            for row in distribution_rows
            if int(row['Foreground']) != 0
        )

    fg_mask_counts = {}
    missing_fg_classes = []
    if fg_mask_root.is_dir():
        for cls_name in required_fg_classes:
            cls_dir = fg_mask_root / cls_name
            count = 0
            if cls_dir.is_dir():
                count = sum(1 for path in cls_dir.iterdir() if path.is_file())
            fg_mask_counts[cls_name] = count
            if count == 0:
                missing_fg_classes.append(cls_name)
    else:
        missing_fg_classes = list(required_fg_classes)

    dtd_images_root = _resolve_dtd_images_root(dtd_root_path)
    dtd_texture_count = 0
    if dtd_images_root is not None:
        for pattern in ('*/*.jpg', '*/*.png', '*.jpg', '*.png'):
            dtd_texture_count += sum(1 for _ in dtd_images_root.glob(pattern))

    report.update({
        'distribution_rows': distribution_rows,
        'required_fg_classes': required_fg_classes,
        'fg_mask_counts': fg_mask_counts,
        'missing_fg_mask_classes': missing_fg_classes,
        'dtd_images_root': str(dtd_images_root) if dtd_images_root is not None else None,
        'dtd_texture_count': dtd_texture_count,
    })
    report['ok'] = (
        report['distribution_exists']
        and report['fg_mask_root_exists']
        and not missing_fg_classes
        and dtd_images_root is not None
        and dtd_texture_count > 0
    )
    return report
