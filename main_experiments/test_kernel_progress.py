"""
Quick test to see kernel progress tracking
"""
import numpy as np
from metrics import calculate_metrics

# Create small test data
print("Creating test data...")
events_est = np.zeros(50000, dtype=[('t', '<f8'), ('x', '<u2'), ('y', '<u2'), ('p', 'i1')])
events_gt = np.zeros(50000, dtype=[('t', '<f8'), ('x', '<u2'), ('y', '<u2'), ('p', 'i1')])

events_est['t'] = np.sort(np.random.uniform(0, 0.1, 50000))
events_est['x'] = np.random.randint(0, 640, 50000)
events_est['y'] = np.random.randint(0, 480, 50000)
events_est['p'] = np.random.choice([-1, 1], 50000)

events_gt['t'] = np.sort(np.random.uniform(0, 0.1, 50000))
events_gt['x'] = np.random.randint(0, 640, 50000)
events_gt['y'] = np.random.randint(0, 480, 50000)
events_gt['p'] = np.random.choice([-1, 1], 50000)

print(f"\nTest data: {len(events_est):,} events each")
print("\nTesting kernel_standard (FINE cubes: 64x48x10ms, ~1000 cubes)...")
print("="*80)

result = calculate_metrics(events_est, events_gt, metric_names=['kernel_standard'])

print("="*80)
print(f"\nResult: {result['kernel_standard']:.6f}")
print("\n✓ Progress tracking test completed!")
