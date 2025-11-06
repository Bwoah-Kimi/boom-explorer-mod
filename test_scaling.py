import os
import sys
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "algo")
)
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "utils")
)

import numpy as np
from algo.custom_dataset import load_dataset, scale_dataset, rescale_dataset

print("=== Testing Dataset Scaling ===\n")

# Load raw dataset
x, y = load_dataset('dataset/converted_dataset.csv', preprocess=False)
print(f"Loaded {len(x)} samples with {x.shape[1]} features")
print(f"Objectives shape: {y.shape} (2 objectives: cycles, cost)\n")

# Original objectives
print("Original Objectives (first 5 samples):")
print("Cycles          | Cost")
for i in range(min(5, len(y))):
    print(f"{y[i, 0]:15.1f} | {y[i, 1]:8.2f}")

# Scaled objectives
y_scaled = scale_dataset(y.copy())
print("\nScaled Objectives (first 5 samples):")
print("Cycles  | Cost")
for i in range(min(5, len(y))):
    print(f"{y_scaled[i, 0]:7.4f} | {y_scaled[i, 1]:.4f}")

# Verify rescaling
y_rescaled = rescale_dataset(y_scaled.copy())
print("\nRescaled Objectives (should match original):")
print("Cycles          | Cost")
for i in range(min(5, len(y))):
    print(f"{y_rescaled[i, 0]:15.1f} | {y_rescaled[i, 1]:8.2f}")

# Check accuracy
max_error = np.max(np.abs(y - y_rescaled))
print(f"\nMax rescaling error: {max_error:.6f}")
if max_error > 1.0:
    print("⚠️  WARNING: Rescaling error is large! Check your scaling parameters.")
else:
    print("✅ Rescaling accuracy is good!")

# Statistics
print("\n=== Scaled Objectives Statistics ===")
print(f"Cycles - Min: {y_scaled[:, 0].min():.4f}, Max: {y_scaled[:, 0].max():.4f}, Mean: {y_scaled[:, 0].mean():.4f}")
print(f"Cost   - Min: {y_scaled[:, 1].min():.4f}, Max: {y_scaled[:, 1].max():.4f}, Mean: {y_scaled[:, 1].mean():.4f}")

print("\n=== Original Objectives Statistics ===")
print(f"Cycles - Min: {y[:, 0].min():.0f}, Max: {y[:, 0].max():.0f}")
print(f"Cost   - Min: {y[:, 1].min():.2f}, Max: {y[:, 1].max():.2f}")

print("\n✅ Scaling test completed!")