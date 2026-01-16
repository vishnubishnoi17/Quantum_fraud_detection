#!/bin/bash

echo "Setting up QML Fraud Classification Project..."

# Create virtual environment
python3 -m venv qml_env
source qml_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p src
mkdir -p results
mkdir -p figures

echo "Setup complete!  Activate environment with: source qml_env/bin/activate"