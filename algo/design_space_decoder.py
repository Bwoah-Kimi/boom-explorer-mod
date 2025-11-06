# Author: baichen318@gmail.com


"""
Design space decoder for converting integer encodings back to actual values.
This is the inverse of the encoding done in convert_json_to_csv.py
"""

import numpy as np
from typing import Dict, List, Union
import torch


# Design space definition (from your original JSON)
DESIGN_SPACE = {
    "num_chiplets": [4, 8, 12, 16],
    "num_arrays": [4, 8, 12, 16],
    "num_DRAM_dies": [2, 3, 4, 6],
    "array_width": [16, 32, 64, 128],
    "array_height": [16, 32, 64, 128],
    "sram_size_MB": [6, 8, 12],
    "dram_type": ["HBM2", "HBM2E", "HBM3", "HBM3E"],
    "d2d_type": ["VLSI22", "VLSI24"],
    "pp": [1, 2, 4],
    "dp": [1]
}

# Parameter names in order (must match the order in CSV encoding)
PARAM_NAMES = [
    "num_chiplets",
    "num_arrays", 
    "num_DRAM_dies",
    "array_width",
    "array_height",
    "sram_size_MB",
    "dram_type",
    "d2d_type",
    "pp",
    "dp"
]


def decode_design_vector(encoded_x: Union[np.ndarray, torch.Tensor, List]) -> Dict:
    """
    Convert encoded design vector to actual parameter values.
    
    Args:
        encoded_x: Array of integers [0, 1, 2, ...] representing encoded parameters
        
    Returns:
        Dictionary with parameter names and actual values
        
    Example:
        encoded_x = [1, 0, 1, 1, 1, 2, 1, 0, 2, 0]
        returns: {
            'num_chiplets': 8,
            'num_arrays': 4,
            'num_DRAM_dies': 3,
            'array_width': 32,
            'array_height': 32,
            'sram_size_MB': 12,
            'dram_type': 'HBM2E',
            'd2d_type': 'VLSI22',
            'pp': 4,
            'dp': 1
        }
    """
    # Convert to numpy array if needed
    if isinstance(encoded_x, torch.Tensor):
        encoded_x = encoded_x.cpu().numpy()
    elif isinstance(encoded_x, list):
        encoded_x = np.array(encoded_x)
    
    # Convert to integers
    encoded_x = encoded_x.astype(int)
    
    # Decode each parameter
    decoded = {}
    for i, param_name in enumerate(PARAM_NAMES):
        idx = encoded_x[i]
        decoded[param_name] = DESIGN_SPACE[param_name][idx]
    
    return decoded


def decode_design_batch(encoded_batch: Union[np.ndarray, torch.Tensor]) -> List[Dict]:
    """
    Decode a batch of design vectors.
    
    Args:
        encoded_batch: Array of shape (n_samples, n_params)
        
    Returns:
        List of decoded design dictionaries
    """
    if isinstance(encoded_batch, torch.Tensor):
        encoded_batch = encoded_batch.cpu().numpy()
    
    decoded_batch = []
    for i in range(encoded_batch.shape[0]):
        decoded_batch.append(decode_design_vector(encoded_batch[i]))
    
    return decoded_batch


def get_design_space_dict() -> Dict:
    """
    Get the design space definition as a dictionary.
    
    Returns:
        Dictionary with parameter names and their possible values
    """
    return DESIGN_SPACE.copy()


def get_param_names() -> List[str]:
    """
    Get the list of parameter names in order.
    
    Returns:
        List of parameter names
    """
    return PARAM_NAMES.copy()
