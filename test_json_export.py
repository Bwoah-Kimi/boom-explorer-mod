#!/usr/bin/env python3
"""
Test the design space decoder and JSON export functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "algo"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

import numpy as np
import torch
from design_space_decoder import (
    decode_design_vector, 
    decode_design_batch,
    get_design_space_dict,
    get_param_names
)
from custom_dataset import rescale_dataset


def test_decoder():
    """Test the design space decoder"""
    print("="*60)
    print("Testing Design Space Decoder")
    print("="*60)
    
    # Test single vector decoding
    print("\n1. Test single vector decoding:")
    encoded = np.array([1, 0, 1, 1, 1, 2, 1, 0, 2, 0])
    decoded = decode_design_vector(encoded)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Test with torch tensor
    print("\n2. Test with torch tensor:")
    encoded_torch = torch.tensor([0, 3, 3, 0, 0, 1, 2, 1, 1, 0])
    decoded = decode_design_vector(encoded_torch)
    print(f"Encoded: {encoded_torch}")
    print(f"Decoded: {decoded}")
    
    # Test batch decoding
    print("\n3. Test batch decoding:")
    encoded_batch = np.array([
        [1, 0, 1, 1, 1, 2, 1, 0, 2, 0],
        [0, 3, 3, 0, 0, 1, 2, 1, 1, 0],
        [2, 2, 2, 2, 2, 0, 0, 0, 0, 0]
    ])
    decoded_batch = decode_design_batch(encoded_batch)
    for i, d in enumerate(decoded_batch):
        print(f"Design {i+1}: {d}")
    
    # Test design space retrieval
    print("\n4. Test design space retrieval:")
    design_space = get_design_space_dict()
    print(f"Design space keys: {list(design_space.keys())}")
    print(f"Number of parameters: {len(design_space)}")
    
    param_names = get_param_names()
    print(f"Parameter names: {param_names}")
    
    print("\n✓ All decoder tests passed!")


def test_rescaling():
    """Test objective rescaling"""
    print("\n" + "="*60)
    print("Testing Objective Rescaling")
    print("="*60)
    
    # Test rescaling scaled objectives back to original
    print("\n1. Test rescaling log-scaled objectives:")
    
    # Example scaled objectives (log-scaled, maximize format)
    scaled_objectives = np.array([7.234, 5.123])
    print(f"Scaled objectives (log): {scaled_objectives}")
    
    unscaled = rescale_dataset(scaled_objectives, cycles_idx=0, cost_idx=1, use_log=True)
    print(f"Unscaled objectives: {unscaled}")
    print(f"  Cycles: {unscaled[0]:,.2f}")
    print(f"  Cost: {unscaled[1]:,.2f}")
    
    # Test with batch
    print("\n2. Test batch rescaling:")
    scaled_batch = np.array([
        [7.234, 5.123],
        [7.890, 4.567],
        [6.543, 5.321]
    ])
    print(f"Scaled batch shape: {scaled_batch.shape}")
    unscaled_batch = rescale_dataset(scaled_batch, cycles_idx=-2, cost_idx=-1, use_log=True)
    print(f"Unscaled batch:")
    for i, obj in enumerate(unscaled_batch):
        print(f"  Design {i+1}: cycles={obj[0]:,.2f}, cost={obj[1]:,.2f}")
    
    print("\n✓ All rescaling tests passed!")


def test_json_structure():
    """Test JSON structure creation"""
    print("\n" + "="*60)
    print("Testing JSON Structure")
    print("="*60)
    
    import json
    from datetime import datetime
    
    # Create example iteration result
    iteration_result = {
        "iteration": 1,
        "timestamp": datetime.now().isoformat(),
        "dse_config": decode_design_vector([1, 0, 1, 1, 1, 2, 1, 0, 2, 0]),
        "objectives": [13659022.8125, 5291.228851803658]
    }
    
    print("\n1. Example iteration result:")
    print(json.dumps(iteration_result, indent=2))
    
    # Create full JSON structure
    json_data = {
        "dse_start_timestamp": datetime.now().isoformat(),
        "design_space": get_design_space_dict(),
        "optimizer_type": "BOOM-Explorer",
        "sampling_algo": "MicroAL + DKL-GP + EHVI",
        "misc_info": {
            "max_bo_steps": 30,
            "mlp_output_dim": 6,
            "total_samples": 1
        },
        "iteration_results": [iteration_result]
    }
    
    print("\n2. Full JSON structure (truncated):")
    print(f"Keys: {list(json_data.keys())}")
    print(f"Design space parameters: {list(json_data['design_space'].keys())}")
    print(f"Number of iteration results: {len(json_data['iteration_results'])}")
    
    # Test JSON serialization
    json_str = json.dumps(json_data, indent=2)
    print(f"\n3. JSON serialization successful")
    print(f"JSON string length: {len(json_str)} characters")
    
    print("\n✓ All JSON structure tests passed!")


if __name__ == "__main__":
    try:
        test_decoder()
        test_rescaling()
        test_json_structure()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe JSON export functionality is ready to use.")
        print("Run your DSE with: python main.py --configs configs/chiplet.yml")
        print("Results will be saved to: rpts/iteration_results.json")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
