#!/bin/bash
# Setup script for main_experiments virtual environment
# This creates a lightweight Python virtual environment without conda

echo "==============================================="
echo "Setting up main_experiments virtual environment"
echo "==============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed or not in PATH"
    exit 1
fi

echo "Python version: $(python3 --version)"

# Create virtual environment in the current directory
ENV_NAME="venv_main_exp"
echo "Creating virtual environment: $ENV_NAME"

python3 -m venv $ENV_NAME

# Check if virtual environment was created successfully
if [ ! -d "$ENV_NAME" ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "Virtual environment created successfully"

# Activate virtual environment
source $ENV_NAME/bin/activate

echo "Activated virtual environment"
echo "Python location: $(which python)"
echo "Pip location: $(which pip)"

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "==============================================="
echo "Setup completed successfully!"
echo "==============================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source $ENV_NAME/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "  deactivate"
echo ""
echo "To test AEDAT4 loading, run:"
echo "  python time_offset_analysis.py --clean_file /path/to/clean.aedat4 --flare_file /path/to/flare.aedat4"
echo ""