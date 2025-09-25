#!/bin/bash

# Prompt2Peptide Setup Script
# One-click setup for Spotlight-ready evaluation

set -e  # Exit on any error

echo "🚀 Setting up Prompt2Peptide for Spotlight submission..."
echo "=================================================="

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version is compatible"
else
    echo "❌ Python $python_version is not compatible. Please install Python 3.8+"
    exit 1
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Install additional packages for Spotlight features
echo "📦 Installing Spotlight-specific packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers
pip install scikit-learn seaborn plotly
pip install jupyter notebook

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p figures results data models logs scripts

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x reproduce.py
chmod +x scripts/*.py

# Download pre-trained models (if needed)
echo "📥 Setting up pre-trained models..."
python -c "
import torch
from transformers import AutoTokenizer, AutoModel
print('Downloading sentence transformer...')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print('✅ Models downloaded successfully')
"

# Run quick test
echo "🧪 Running quick test..."
python -c "
import sys
sys.path.append('.')
from motifgen.generate import generate
print('Testing basic generation...')
result = generate('test peptide', n=1)
print(f'✅ Generated: {result[0] if result else \"No result\"}')
"

echo ""
echo "🎉 Setup completed successfully!"
echo "=================================================="
echo ""
echo "📋 Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Run evaluation: python reproduce.py"
echo "3. Or use Docker: docker-compose up"
echo ""
echo "🚀 Ready for Spotlight submission!"
echo ""
echo "📁 Project structure:"
echo "├── paper/           # LaTeX paper and figures"
echo "├── motifgen/        # Core Python package"
echo "├── scripts/         # Evaluation scripts"
echo "├── data/            # Results and data"
echo "├── figures/         # Generated figures"
echo "└── reproduce.py     # One-click reproduction"
echo ""
echo "🎯 Target venues:"
echo "• AAAI FMs4Bio (Foundation Models for Biological Discoveries)"
echo "• NeurIPS MLSB (Machine Learning in Structural Biology)"
echo "• ICML WCB (Workshop on Computational Biology)"
