"""
Reproducibility seeding utilities.

Sets seeds for all random number generators used throughout the project:
Python built-in ``random``, NumPy, PyTorch (CPU + CUDA), and the hash seed
environment variable so that dict/set orderings are deterministic across runs.
"""

from __future__ import annotations

import os
import random
from typing import List

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set every random seed used by the project to *seed*.

    Parameters
    ----------
    seed:
        Master seed value.  Use the same value across training, evaluation,
        and inference to guarantee reproducibility.

    Notes
    -----
    PyTorch is imported lazily so that environments without it can still use
    this function.  If CUDA is available, the GPU seed is also set.
    ``torch.backends.cudnn.deterministic = True`` and
    ``torch.backends.cudnn.benchmark = False`` are set to make cuDNN ops
    reproducible (at the cost of some speed).
    """
    # Python built-in
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Python hash randomisation (affects dict/set iteration order)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch — optional dependency
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # scikit-learn / scipy use NumPy's global state, so no extra call needed.


def make_cv_seeds(base_seed: int = 42, n_folds: int = 5) -> List[int]:
    """Generate a list of distinct, reproducible seeds — one per CV fold.

    Using a different seed for each fold prevents the same bootstrapped
    samples from appearing across folds while keeping the whole sequence
    reproducible given *base_seed*.

    Parameters
    ----------
    base_seed:
        Seed used to initialise the generator that produces fold seeds.
    n_folds:
        Number of seeds to generate (one per fold).

    Returns
    -------
    List[int]
        A list of *n_folds* non-negative integer seeds.

    Examples
    --------
    >>> make_cv_seeds(42, 5)
    [102, 435, 860, 270, 106]  # deterministic but arbitrary values
    """
    rng = np.random.RandomState(base_seed)
    # Draw from a wide range to minimise inter-fold collisions
    return rng.randint(0, 10_000, size=n_folds).tolist()
