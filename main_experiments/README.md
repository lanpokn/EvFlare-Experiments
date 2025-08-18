# Event Glare Removal - Evaluation Framework

This framework provides a modular approach for evaluating **pre-processed** glare removal results against ground truth data. The design is completely flexible regarding file locations and formats, making it suitable for various experimental setups.

**Important**: This framework does NOT perform glare removal processing. It evaluates results that have already been processed by external methods.

## Framework Overview

The framework consists of four core components:

1. **`data_loader.py`** - Handles loading event data from various formats (.aedat4, .h5, .npy)
2. **`methods.py`** - Defines method result loaders for pre-processed data
3. **`metrics.py`** - Implements evaluation metrics (Chamfer Distance, Gaussian Distance, etc.)
4. **`evaluator.py`** - Orchestrates the complete evaluation pipeline
5. **`run_main_experiment.py`** - Example usage and main execution script

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Method Results and Ground Truth Data

Edit `create_example_method_results()` in `run_main_experiment.py` to point to your processed results:

```python
method_results = [
    MethodResult(
        name="MyAwesomeModel_v1", 
        result_file_path="path/to/your/processed_result.aedat4",
        model_version="1.0"
    ),
    MethodResult(
        name="CompetitorMethod_A", 
        result_file_path="path/to/competitor_result.h5",
        paper_reference="Smith et al. 2024"
    ),
    # Add more method results...
]
```

Edit `create_example_evaluation_samples()` to point to your ground truth data:

```python
evaluation_samples = [
    {
        'name': 'sample_01',
        'gt': 'path/to/your/ground_truth_data.aedat4'  # Clean background only
    },
    # Add more samples...
]
```

### 3. Run Experiment

```bash
python run_main_experiment.py --output results/
```

## Data Format

All event data must be NumPy structured arrays with fields:
- `'t'` (float64): Timestamp in microseconds
- `'x'` (uint16): X coordinate
- `'y'` (uint16): Y coordinate  
- `'p'` (int8): Polarity (-1 or 1, or 0 or 1)

## Supported File Formats

- **AEDAT4** (.aedat4): DVS camera format (requires `dv-processing` library)
- **HDF5** (.h5): Custom HDF5 format (requires `h5py`)
- **NumPy** (.npy): Direct NumPy array storage

## Architecture Principles

### Modular Design
Each component has a single responsibility and clear interfaces, allowing easy extension and modification.

### Format Agnostic
The framework doesn't assume any specific directory structure or file naming conventions. You provide explicit file paths.

### Result-Based Evaluation
Designed to evaluate pre-processed results from any glare removal method, regardless of implementation details.

### Comprehensive Metrics
Built-in support for multiple evaluation metrics, with easy extension for custom metrics.

## Example Workflow

```python
# 1. Define method results to evaluate
method_results = [
    MethodResult(name="MyAwesomeModel", result_file_path="results/my_model_output.aedat4"),
    MethodResult(name="CompetitorA", result_file_path="results/competitor_a_output.h5"),
    MethodResult(name="CompetitorB", result_file_path="results/competitor_b_output.npy")
]

# 2. Define evaluation samples (ground truth data)
samples = [
    {'name': 'scene1', 'gt': 'ground_truth/scene1_clean.h5'},
    {'name': 'scene2', 'gt': 'ground_truth/scene2_clean.aedat4'}
]

# 3. Run evaluation
evaluator = MainExperimentEvaluator(method_results)
evaluator.run_batch_evaluation(samples, save_intermediate=True, output_dir="results/")

# 4. Get results
df = evaluator.get_results_as_dataframe()
evaluator.save_results_to_csv("experiment_results.csv")
```

## Current Implementation Status

### ✅ Completed
- Core framework structure
- Abstract interfaces for all components
- Metrics implementation (Chamfer Distance, Gaussian Distance from user specification)
- Evaluation pipeline orchestration
- Results collection and export
- Method result loading system

### 🚧 Needs Implementation
- **AEDAT4 file loading** (requires dv-processing library integration)
- **HDF5 file loading** (requires h5py integration and knowledge of your file structure) 

### 💡 Future Extensions
- Configuration file support for batch experiments
- Visualization tools for results comparison
- Advanced statistical analysis
- Performance benchmarking tools

## File Structure

```
main_experiments/
├── data_loader.py          # Event data loading abstraction
├── methods.py              # Method result loaders for pre-processed data  
├── metrics.py              # Evaluation metrics implementation
├── evaluator.py            # Main evaluation orchestrator
├── run_main_experiment.py  # Example usage and main script
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── [your_data.zip]        # Your actual data files
```

## Notes for Implementation

1. **Data Loading**: The framework supports multiple formats but needs actual file reading code for .aedat4 and .h5 files if you use those formats.

2. **Method Results**: Simply point `MethodResult` objects to your pre-processed output files from any glare removal method.

3. **Metrics**: The provided metrics (Chamfer Distance, Gaussian Distance) use the exact code you provided, with additional utility metrics.

4. **Flexibility**: The framework is designed to work with any file paths and doesn't enforce any specific directory structure.

5. **Processing Pipeline**: This framework assumes all glare removal processing is done externally. It only handles evaluation of results.

This framework provides a clean, focused evaluation system that can immediately be used with your processed glare removal results.