# Author: baichen318@gmail.com


import csv
import torch
import numpy as np
from utils import if_exist
from typing import Union, Tuple, List


def load_dataset(path: str, preprocess=True) -> Tuple[np.ndarray, np.ndarray]:
    def _read_csv() -> Tuple[List[List[int]], List[str]]:
        dataset = []
        if_exist(path, strict=True)
        with open(path, 'r') as f:
            reader = csv.reader(f)
            title = next(reader)
            for row in reader:
                dataset.append(row)
        return dataset, title

    def validate(dataset: List[List[int]]) -> np.ndarray:
        """
            `dataset`: <tuple>
        """
        data = []
        for item in dataset:
            _data = []
            f = item[0].split(' ')
            for i in f:
                _data.append(int(i))
            for i in item[1:]:
                _data.append(float(i))
            data.append(_data)
        data = np.array(data)

        return data

    dataset, _ = _read_csv()
    dataset = validate(dataset)
    if preprocess:
        dataset = scale_dataset(dataset)
    # split to two matrices
    x = []
    y = []
    for data in dataset:
        # Handle 2 objectives (cycles, cost) - no time column
        x.append(data[:-2])
        y.append(np.array([data[-2], data[-1]]))
    return np.array(x), np.array(y)


def scale_dataset(
    dataset: Union[torch.Tensor, np.ndarray],
    cycles_idx: int = -2,
    cost_idx: int = -1,
    use_log: bool = True
) -> Union[torch.Tensor, np.ndarray]:
    """
    Scale objectives from "minimize" to "maximize" format for Bayesian Optimization.
    
    For LONG-TAIL distributions, we use LOG scaling to better spread values:
    - Most values near minimum get reasonable spacing
    - Extreme outliers don't dominate the scale
    - Scaled values can be > 2 (that's OK!)
    
    Two scaling modes:
    1. LOG SCALING (recommended for long-tail): log(max + 1 - x)
       - Better for skewed distributions
       - Spreads out small values more
    
    2. LINEAR SCALING: (max - x) / scale
       - Better for uniform distributions
       - Compresses most values if long-tail
    
    After running analyze_converted_dataset.py:
    - If median << mean: use log scaling (long-tail)
    - If median ≈ mean: use linear scaling (uniform)
    """
    if isinstance(dataset, torch.Tensor):
        _dataset = dataset.clone()
    else:
        _dataset = dataset.copy()
    
    if use_log:
        # ===== LOG SCALING for LONG-TAIL distributions =====
        # Formula: log(max + 1 - x)
        # This compresses the tail while expanding the bulk of values
        
        # For cycles (minimize -> maximize via log)
        CYCLES_MAX = 450000000  # Above your max
        _dataset[:, cycles_idx] = np.log10(CYCLES_MAX + 1 - _dataset[:, cycles_idx])
        
        # For cost (minimize -> maximize via log)
        COST_MAX = 270000  # Above your max
        _dataset[:, cost_idx] = np.log10(COST_MAX + 1 - _dataset[:, cost_idx])
        
    else:
        # ===== LINEAR SCALING (for reference) =====
        # Formula: (max - x) / scale
        
        CYCLES_MAX = 450000000
        CYCLES_SCALE = 220000000
        _dataset[:, cycles_idx] = (CYCLES_MAX - _dataset[:, cycles_idx]) / CYCLES_SCALE
        
        COST_MAX = 270000
        COST_SCALE = 5000
        _dataset[:, cost_idx] = (COST_MAX - _dataset[:, cost_idx]) / COST_SCALE
    
    return _dataset


def rescale_dataset(
    dataset: Union[torch.Tensor, np.ndarray],
    cycles_idx: int = -2,
    cost_idx: int = -1,
    use_log: bool = True
) -> Union[torch.Tensor, np.ndarray]:
    """
    Inverse of scale_dataset() - converts scaled values back to original objectives.
    
    IMPORTANT: use_log MUST match the scale_dataset() setting!
    """
    if isinstance(dataset, torch.Tensor):
        _dataset = dataset.clone()
    else:
        _dataset = dataset.copy()
    
    # Handle both 1D and 2D output formats
    if _dataset.ndim == 1:
        if use_log:
            # Reverse log scaling: x = max + 1 - 10^(scaled)
            temp = np.zeros(2)
            temp[0] = 450000000 + 1 - (10 ** _dataset[0])
            temp[1] = 270000 + 1 - (10 ** _dataset[1])
        else:
            # Reverse linear scaling
            temp = np.zeros(2)
            temp[0] = 450000000 - (_dataset[0] * 220000000)
            temp[1] = 2700000 - (_dataset[1] * 5000)
        return temp
    
    elif _dataset.shape[1] == 2:
        cycles_idx = -2
        cost_idx = -1
    else:
        raise ValueError(f"Expected 2 columns, got {_dataset.shape[1]}")
    
    if use_log:
        # Reverse log scaling: x = max + 1 - 10^(scaled)
        _dataset[:, cycles_idx] = 450000000 + 1 - (10 ** _dataset[:, cycles_idx])
        _dataset[:, cost_idx] = 270000 + 1 - (10 ** _dataset[:, cost_idx])
    else:
        # Reverse linear scaling: x = max - (scaled * scale)
        _dataset[:, cycles_idx] = 450000000 - (_dataset[:, cycles_idx] * 220000000)
        _dataset[:, cost_idx] = 270000 - (_dataset[:, cost_idx] * 5000)

    return _dataset


def tensor_to_ndarray(tensor: torch.Tensor) -> np.ndarray:
    return tensor.numpy()


def ndarray_to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.Tensor(array)