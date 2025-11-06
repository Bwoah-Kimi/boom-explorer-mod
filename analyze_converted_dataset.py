import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load YOUR converted dataset
df = pd.read_csv('dataset/converted_dataset.csv')

# Extract objectives
total_cycles = df['total_cycles'].values
total_cost = df['total_cost'].values

print("=== YOUR Dataset Statistics ===\n")

print("Total Cycles:")
print(f"  Min: {total_cycles.min():.2f}")
print(f"  Max: {total_cycles.max():.2f}")
print(f"  Mean: {total_cycles.mean():.2f}")
print(f"  Median: {np.median(total_cycles):.2f}")
print(f"  Std: {total_cycles.std():.2f}")
print(f"  25th percentile: {np.percentile(total_cycles, 25):.2f}")
print(f"  75th percentile: {np.percentile(total_cycles, 75):.2f}")
print(f"  95th percentile: {np.percentile(total_cycles, 95):.2f}")
print(f"  99th percentile: {np.percentile(total_cycles, 99):.2f}\n")

print("Total Cost:")
print(f"  Min: {total_cost.min():.2f}")
print(f"  Max: {total_cost.max():.2f}")
print(f"  Mean: {total_cost.mean():.2f}")
print(f"  Median: {np.median(total_cost):.2f}")
print(f"  Std: {total_cost.std():.2f}")
print(f"  25th percentile: {np.percentile(total_cost, 25):.2f}")
print(f"  75th percentile: {np.percentile(total_cost, 75):.2f}")
print(f"  95th percentile: {np.percentile(total_cost, 95):.2f}")
print(f"  99th percentile: {np.percentile(total_cost, 99):.2f}\n")

# Detect long-tail distribution
def is_long_tail(values):
    mean = np.mean(values)
    median = np.median(values)
    # If mean >> median, it's right-skewed (long right tail)
    skew_ratio = mean / median
    return skew_ratio > 1.5, skew_ratio

cycles_is_long_tail, cycles_skew = is_long_tail(total_cycles)
cost_is_long_tail, cost_skew = is_long_tail(total_cost)

print("=== Distribution Analysis ===\n")
print(f"Cycles: {'LONG-TAIL' if cycles_is_long_tail else 'UNIFORM-ISH'} (mean/median = {cycles_skew:.2f})")
print(f"Cost:   {'LONG-TAIL' if cost_is_long_tail else 'UNIFORM-ISH'} (mean/median = {cost_skew:.2f})\n")

if cycles_is_long_tail or cost_is_long_tail:
    print("⚠️  RECOMMENDATION: Use LOG SCALING (use_log=True)")
    print("   Most values are near minimum, log will spread them out better.\n")
else:
    print("✅ RECOMMENDATION: Use LINEAR SCALING (use_log=False)")
    print("   Values are fairly uniform, linear scaling is fine.\n")

# Show scaling results
print("=== Scaling Examples (LOG) ===\n")
cycles_max = 450000000
cost_max = 270000

print("Cycles (first 5 samples):")
print("Original -> Log Scaled")
for i in range(min(5, len(total_cycles))):
    scaled = np.log10(cycles_max + 1 - total_cycles[i])
    print(f"{total_cycles[i]:12.0f} -> {scaled:8.4f}")

print("\nCost (first 5 samples):")
print("Original -> Log Scaled")
for i in range(min(5, len(total_cost))):
    scaled = np.log10(cost_max + 1 - total_cost[i])
    print(f"{total_cost[i]:8.2f} -> {scaled:8.4f}")

# Show range of scaled values
all_cycles_scaled = np.log10(cycles_max + 1 - total_cycles)
all_cost_scaled = np.log10(cost_max + 1 - total_cost)

print("\n=== Scaled Value Ranges (LOG) ===")
print(f"Cycles: [{all_cycles_scaled.min():.4f}, {all_cycles_scaled.max():.4f}]")
print(f"Cost:   [{all_cost_scaled.min():.4f}, {all_cost_scaled.max():.4f}]")

# Optional: plot histograms if matplotlib available
try:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].hist(total_cycles, bins=50, edgecolor='black')
    axes[0, 0].set_title('Original Cycles Distribution')
    axes[0, 0].set_xlabel('Cycles')
    
    axes[0, 1].hist(all_cycles_scaled, bins=50, edgecolor='black')
    axes[0, 1].set_title('Log-Scaled Cycles Distribution')
    axes[0, 1].set_xlabel('Scaled Value')
    
    axes[1, 0].hist(total_cost, bins=50, edgecolor='black')
    axes[1, 0].set_title('Original Cost Distribution')
    axes[1, 0].set_xlabel('Cost')
    
    axes[1, 1].hist(all_cost_scaled, bins=50, edgecolor='black')
    axes[1, 1].set_title('Log-Scaled Cost Distribution')
    axes[1, 1].set_xlabel('Scaled Value')
    
    plt.tight_layout()
    plt.savefig('dataset_distribution_analysis.png', dpi=150)
    print("\n📊 Saved distribution plots to: dataset_distribution_analysis.png")
except ImportError:
    print("\n(matplotlib not available, skipping plots)")