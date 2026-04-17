"""FrEIA compatibility and performance helpers."""

from __future__ import annotations

import numpy as np
import torch


def _sample_special_ortho_torch(dim: int, device: str | None = None) -> np.ndarray:
    """Sample an ``SO(dim)`` matrix with GPU QR when CUDA is available."""
    target_device = device or 'cuda'
    matrix = torch.randn(dim, dim, device=target_device, dtype=torch.float32)
    q, r = torch.linalg.qr(matrix, mode='reduced')

    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    q = q * signs

    if torch.linalg.det(q).item() < 0:
        q[:, 0].neg_()

    return q.cpu().numpy()


def patch_freia_soft_permutation_rvs(min_dim: int = 512, device: str | None = None) -> None:
    """Patch FrEIA soft-permutation init to avoid slow SciPy SO(n) sampling.

    FrEIA's default path uses ``scipy.stats.special_ortho_group.rvs(dim)``,
    which becomes prohibitively slow around 1024 channels. On CUDA hosts we can
    generate the same kind of orthogonal matrix with a GPU QR decomposition and
    hand the result back to FrEIA as a NumPy array.
    """
    import FrEIA.modules.all_in_one_block as all_in_one_block

    current_rvs = all_in_one_block.special_ortho_group.rvs
    original_rvs = getattr(current_rvs, '_baoiad_original_rvs', current_rvs)

    def _patched_rvs(dim: int, *args, **kwargs):
        if torch.cuda.is_available() and dim >= min_dim:
            return _sample_special_ortho_torch(dim, device=device)
        return original_rvs(dim, *args, **kwargs)

    _patched_rvs._baoiad_fast_soft_perm = True  # type: ignore[attr-defined]
    _patched_rvs._baoiad_original_rvs = original_rvs  # type: ignore[attr-defined]
    all_in_one_block.special_ortho_group.rvs = _patched_rvs


__all__ = ['patch_freia_soft_permutation_rvs']
