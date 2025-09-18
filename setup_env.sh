#!/bin/bash
set -e

# Setup script for Cuban Women Breast Cancer Risk Analysis project
# Compatible with GitHub Codespaces and Termux

echo "===== Setting up environment for Breast Cancer Risk Analysis Project ====="

# Detect environment
if [ -n "$CODESPACES" ]; then
    echo "Detected GitHub Codespaces environment"
    ENV_TYPE="codespaces"
elif [ -n "$(command -v termux-info 2>/dev/null)" ]; then
    echo "Detected Termux environment"
    ENV_TYPE="termux"
else
    echo "Standard Linux/macOS environment detected"
    ENV_TYPE="standard"
fi

# Setup Python virtual environment
if [ "$ENV_TYPE" = "termux" ]; then
    echo "Installing required packages in Termux..."
    pkg update -y
    pkg install -y python clang libffi libandroid-spawn git
    
    # Fix potential issues with Numpy on Termux
    export MATHLIB="m"
    export CFLAGS="-Wno-error=implicit-function-declaration"
    
    # Create virtual environment
    echo "Creating Python virtual environment..."
    python -m venv venv
    source venv/bin/activate
else
    # Codespaces or standard Linux/macOS
    echo "Creating Python virtual environment..."
    python -m venv venv
    source venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo "Installing project dependencies..."
pip install -r requirements.txt

# Install development dependencies if needed
if [ "$1" = "--dev" ]; then
    echo "Installing development dependencies..."
    pip install black isort mypy pytest pytest-cov
fi

# Create project directories if they don't exist
echo "Creating project directories..."
mkdir -p output/{models,artifacts,svgs,reports,logs,data_raw,data_processed}

# Create a .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "# Environment variables for the project" > .env
    echo "PYTHONPATH=$(pwd)" >> .env
    echo "SEED=42" >> .env
    echo "VERBOSE=1" >> .env
fi

# Set up pre-commit hooks if Git is available
if [ -d .git ] && [ "$1" = "--dev" ]; then
    echo "Setting up pre-commit hooks..."
    pip install pre-commit
    cat > .pre-commit-config.yaml << EOF
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    -   id: isort
-   repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
    -   id: black
EOF
    pre-commit install
fi

echo "===== Environment setup complete! ====="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To train a model:"
echo "  python training.py --data \"Dataset 1- Breast cancer risk factors in Cuban women Dataset (1).csv\" --meta \"Dataset 1 - Breast cancer risk factors in Cuban women Dataset Description Detail.txt\" --outdir output --model xgboost --cv 5 --trials 50 --seed 42 --verbose"
echo ""
echo "To run inference:"
echo "  python main.py --model output/models/model_best.joblib --data new_patients.csv --outdir output_run --predict --svgs output_run/svgs --eval"
echo ""
echo "===== Happy analyzing! ====="