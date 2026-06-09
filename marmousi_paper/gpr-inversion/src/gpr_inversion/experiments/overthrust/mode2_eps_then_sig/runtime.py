"""Runtime utilities shared by migrated inversion scripts."""

import os
import random

import numpy as np
import torch


def same_seeds(seed):
    """Set common random seeds so repeated runs are easier to compare."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_unique_dir(target_dir):
    """Create a result directory without deleting older runs.

    If ``target_dir`` already exists, a suffix such as ``_run001`` is used. This
    matches the project safety rule: old result folders are preserved.
    """
    candidate = target_dir
    if os.path.exists(candidate):
        base = target_dir
        run_idx = 1
        while os.path.exists(candidate):
            candidate = f"{base}_run{run_idx:03d}"
            run_idx += 1
    os.makedirs(candidate, exist_ok=True)
    return candidate


def get_parameter_number(net):
    """Return total and trainable parameter counts for a torch module."""
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


def sanitize_inversion_gradient(grad, fixed_top_rows):
    """Mask the fixed shallow layer and normalize/clamp the inversion gradient."""
    if torch.isnan(grad).any() or torch.isinf(grad).any():
        print("警告：梯度包含nan或inf值！")
        grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)

    grad[:fixed_top_rows, :] = 0
    max_abs_value = torch.max(torch.abs(grad)).item()
    if max_abs_value != 0:
        grad /= max_abs_value

    return torch.clamp(grad, -1.0, 1.0)
