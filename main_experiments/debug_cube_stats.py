"""
Debug script to analyze cube event distribution
"""
import numpy as np
import h5py
from metrics import events_to_spike_cubes

# Load a real sample
h5_file = "/mnt/e/BaiduSyncdisk/2025/event_flick_flare/datasets/simu_flare_removal/target/composed_00000_bg_light.h5"

print(f"Loading: {h5_file}")
with h5py.File(h5_file, 'r') as f:
    events = np.zeros(len(f['events/t']), dtype=[('t', '<f8'), ('x', '<u2'), ('y', '<u2'), ('p', 'i1')])
    events['t'] = f['events/t'][:] / 1e6  # Convert to seconds
    events['x'] = f['events/x'][:]
    events['y'] = f['events/y'][:]
    events['p'] = np.where(f['events/p'][:] == 1, 1, -1)

print(f"Total events: {len(events):,}")
print(f"Duration: {events['t'][-1] - events['t'][0]:.3f}s")

# Convert to float format
evs_float = np.zeros((4, events.shape[0]), dtype=np.float64)
evs_float[0, :] = events['p']
evs_float[1, :] = events['t']
evs_float[2, :] = events['x']
evs_float[3, :] = events['y']

# Test different cube configurations
configs = [
    ("ULTRA COARSE (kernel_standard)", 160, 120, 50000),
    ("COARSE (kernel_fine)", 80, 60, 25000),
    ("ULTRA COARSE spatial", 128, 96, 100000),
    ("ULTRA COARSE temporal", 320, 240, 20000),
]

for name, x_cube, y_cube, t_cube in configs:
    print(f"\n{'='*70}")
    print(f"Configuration: {name}")
    print(f"Cube size: {x_cube}×{y_cube}×{t_cube/1000:.0f}ms")

    cubes = events_to_spike_cubes(evs_float, 640, 480, x_cube, y_cube, t_cube)

    # Analyze cube statistics
    non_empty_cubes = [c for c in cubes if len(c) > 0]
    cube_sizes = [len(c) for c in non_empty_cubes]

    print(f"Total cubes: {len(cubes)}")
    print(f"Non-empty cubes: {len(non_empty_cubes)}")

    if cube_sizes:
        print(f"Events per non-empty cube:")
        print(f"  Min:    {min(cube_sizes):,}")
        print(f"  Max:    {max(cube_sizes):,}")
        print(f"  Mean:   {np.mean(cube_sizes):,.0f}")
        print(f"  Median: {int(np.median(cube_sizes)):,}")

        # Estimate computation for worst-case cube
        max_size = max(cube_sizes)
        worst_case_ops = 3 * max_size**2  # K(e1,e1) + K(e2,e2) + 2*K(e1,e2)
        print(f"\nWORST CASE CUBE:")
        print(f"  Events: {max_size:,}")
        print(f"  Operations: {worst_case_ops:,.0f} (= 3 × {max_size:,}²)")
        print(f"  Matrix size per kernel call: {max_size:,} × {max_size:,} = {max_size**2:,.0f} elements")

        # Check if chunking will help
        chunk_size = 1000
        chunks_needed = int(np.ceil(max_size / chunk_size))**2
        print(f"  Chunks needed (chunk_size={chunk_size}): {chunks_needed:,}")
        print(f"  Memory per chunk: {chunk_size**2 * 8 / 1e6:.1f} MB")
