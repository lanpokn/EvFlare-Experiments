# Kernel Metric Usage Guide

## Quick Start

### Test single sample with default settings
```bash
python evaluate_all_methods.py --metrics kernel_standard --num-samples 1 --output results
```

## Kernel Sampling Configuration

### Command Line Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `--kernel-sampling` | on/off | on | Enable/disable sampling |
| `--kernel-max-events` | integer | 10000 | Max events per cube before sampling |
| `--kernel-cube-scale` | float | 1.0 | Cube resolution scale (2.0=coarser, 0.5=finer) |
| `--kernel-verbose` | on/off | on | Show progress tracking |

## Usage Examples

### 1. Default: Conservative sampling (RECOMMENDED)
```bash
# Default: sampling enabled, max_events=10000, verbose
python evaluate_all_methods.py \
  --metrics kernel_standard kernel_fine kernel_spatial kernel_temporal \
  --num-samples 1 \
  --output results_kernel
```

### 2. Disable sampling (SLOW but exact)
```bash
# No sampling - mathematically exact but VERY SLOW
python evaluate_all_methods.py \
  --metrics kernel_standard \
  --num-samples 1 \
  --kernel-sampling off \
  --output results_kernel
```

### 3. Aggressive sampling (FAST but less accurate)
```bash
# Max 5000 events per cube - faster but ~5% error
python evaluate_all_methods.py \
  --metrics kernel_fine \
  --num-samples 1 \
  --kernel-sampling on \
  --kernel-max-events 5000 \
  --output results_kernel
```

### 4. Silent mode (no progress tracking)
```bash
# Disable progress bars for batch processing
python evaluate_all_methods.py \
  --metrics kernel_standard \
  --num-samples 1 \
  --kernel-verbose off \
  --output results_kernel
```

### 5. Test cube scale (resolution)
```bash
# Test 1: Default scale (1.0) - baseline
python evaluate_all_methods.py \
  --metrics kernel_fine \
  --num-samples 1 \
  --kernel-cube-scale 1.0 \
  --output results_scale_1.0

# Test 2: Coarser cubes (2.0) - fewer cubes, slower per cube
python evaluate_all_methods.py \
  --metrics kernel_fine \
  --num-samples 1 \
  --kernel-cube-scale 2.0 \
  --output results_scale_2.0

# Test 3: Finer cubes (0.5) - more cubes, faster per cube
python evaluate_all_methods.py \
  --metrics kernel_fine \
  --num-samples 1 \
  --kernel-cube-scale 0.5 \
  --output results_scale_0.5

# Test 4: Ultra fine cubes (0.25) - many cubes, very fast per cube
python evaluate_all_methods.py \
  --metrics kernel_fine \
  --num-samples 1 \
  --kernel-cube-scale 0.25 \
  --output results_scale_0.25
```

### 6. Test sampling thresholds
```bash
# Test 1: No sampling (baseline, slowest)
python evaluate_all_methods.py \
  --metrics kernel_fine \
  --num-samples 1 \
  --kernel-sampling off \
  --output results_kernel_exact

# Test 2: max_events=20000 (conservative)
python evaluate_all_methods.py \
  --metrics kernel_fine \
  --num-samples 1 \
  --kernel-max-events 20000 \
  --output results_kernel_20k

# Test 3: max_events=10000 (default)
python evaluate_all_methods.py \
  --metrics kernel_fine \
  --num-samples 1 \
  --kernel-max-events 10000 \
  --output results_kernel_10k

# Test 4: max_events=5000 (aggressive)
python evaluate_all_methods.py \
  --metrics kernel_fine \
  --num-samples 1 \
  --kernel-max-events 5000 \
  --output results_kernel_5k
```

### 7. Combined optimization (fastest)
```bash
# Ultra aggressive: fine cubes + aggressive sampling
python evaluate_all_methods.py \
  --metrics kernel_fine \
  --num-samples 1 \
  --kernel-cube-scale 0.5 \
  --kernel-max-events 5000 \
  --output results_ultra_fast
```

## Understanding the Output

### Progress Tracking (when --kernel-verbose on)
```
[Kernel Config] Sampling: True, Max events: 10000, Verbose: True

  [Kernel] Total cubes: 30720, Cube splitting: 0.15s
  [Kernel] Events: est=1,730,000, gt=1,730,000
  [Kernel] Progress: 10000/30720 (32.6%) | Non-empty: 8542 | Speed: 125.3 cubes/s | ETA: 165s
  [Kernel] Progress: 20000/30720 (65.1%) | Non-empty: 17023 | Speed: 132.1 cubes/s | ETA: 81s
  [Kernel] Progress: 30720/30720 (100.0%) | Non-empty: 26145 | Speed: 128.7 cubes/s | ETA: 0s
  [Kernel] ✓ Completed in 245.3s (split: 0.15s, compute: 245.2s)
  [Kernel] Non-empty cubes: 26145/30720 (85.1%)
```

**Key metrics:**
- **Speed**: cubes/s - higher is better
- **ETA**: estimated time remaining
- **Non-empty ratio**: % of cubes with events (indicates event distribution)

## Performance Guidelines

### Cube Configurations (Current)

| Variant | Cube Size | Total Cubes | Avg Events/Cube | Relative Speed |
|---------|-----------|-------------|-----------------|----------------|
| kernel_fine | 20×20×2.5ms | ~30K | 56 | ⚡⚡⚡ Fastest |
| kernel_standard | 40×30×5ms | ~5K | 338 | ⚡⚡ Fast |
| kernel_spatial | 32×24×10ms | ~4K | 432 | ⚡⚡ Fast |
| kernel_temporal | 80×60×2.5ms | ~2.6K | 676 | ⚡ Moderate |

### Sampling Impact

| Max Events | Speed | Accuracy | Use Case |
|------------|-------|----------|----------|
| off (no sampling) | 1x | 100% | Baseline/validation |
| 20000 | ~2x | 99% | High accuracy |
| 10000 (default) | ~4x | 97% | Recommended balance |
| 5000 | ~16x | 95% | Quick testing |
| 2000 | ~64x | 90% | Rough estimation |

## Recommended Workflow

1. **Quick test** (1 sample, default settings):
```bash
python evaluate_all_methods.py --metrics kernel_fine --num-samples 1 --output results
```

2. **Find optimal settings** (test different max_events):
```bash
# Try 5000, 10000, 20000 and compare speed vs accuracy
```

3. **Full evaluation** (all samples, optimized settings):
```bash
python evaluate_all_methods.py \
  --metrics kernel_standard kernel_fine kernel_spatial kernel_temporal \
  --kernel-max-events 10000 \
  --output results_kernel
```
