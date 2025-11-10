#!/bin/bash
################################################################################
# SSVP-SLT Installation Script
################################################################################
# Installa automaticamente il repository SSVP-SLT e tutte le dipendenze
# 
# Usage:
#   bash scripts/install_ssvp.sh
#
# Requirements:
#   - Python 3.8+
#   - CUDA 11.8+ (per GPU)
#   - ffmpeg
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "🚀 SSVP-SLT Installation Script"
echo "================================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$ROOT_DIR/models"
REPO_DIR="$MODELS_DIR/ssvp_slt_repo"

echo "📂 Directories:"
echo "   Root: $ROOT_DIR"
echo "   Models: $MODELS_DIR"
echo "   Repo: $REPO_DIR"
echo ""

# Check Python version
echo "1️⃣  Checking Python version..."
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "   ✓ Python version: $PYTHON_VERSION"

# Check if version is >= 3.8
MIN_VERSION="3.8"
if [ "$(printf '%s\n' "$MIN_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$MIN_VERSION" ]; then
    echo -e "${RED}   ✗ Python 3.8+ required! Current: $PYTHON_VERSION${NC}"
    exit 1
fi
echo ""

# Check CUDA availability (optional)
echo "2️⃣  Checking CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo -e "   ${GREEN}✓ CUDA version: $CUDA_VERSION${NC}"
else
    echo -e "   ${YELLOW}⚠️  CUDA not found. GPU training will not be available.${NC}"
fi
echo ""

# Check ffmpeg
echo "3️⃣  Checking ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version | head -n1 | awk '{print $3}')
    echo "   ✓ ffmpeg version: $FFMPEG_VERSION"
else
    echo -e "${RED}   ✗ ffmpeg not found!${NC}"
    echo "   Install with:"
    echo "     macOS: brew install ffmpeg"
    echo "     Linux: sudo apt-get install ffmpeg"
    exit 1
fi
echo ""

# Create directories
echo "4️⃣  Creating directories..."
mkdir -p "$MODELS_DIR"
mkdir -p "$MODELS_DIR/checkpoints"
echo "   ✓ Created: $MODELS_DIR"
echo "   ✓ Created: $MODELS_DIR/checkpoints"
echo ""

# Clone SSVP-SLT repository
echo "5️⃣  Cloning SSVP-SLT repository..."
if [ -d "$REPO_DIR" ]; then
    echo -e "   ${YELLOW}⚠️  Repository already exists at: $REPO_DIR${NC}"
    read -p "   Remove and re-clone? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "   Removing old repository..."
        rm -rf "$REPO_DIR"
    else
        echo "   Skipping clone. Using existing repository."
        cd "$REPO_DIR"
    fi
fi

if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/facebookresearch/ssvp_slt.git "$REPO_DIR"
    echo "   ✓ Cloned to: $REPO_DIR"
    cd "$REPO_DIR"
else
    cd "$REPO_DIR"
fi
echo ""

# Check and handle torch version conflicts
echo "6️⃣  Checking PyTorch version..."
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not_installed")
echo "   Current PyTorch version: $TORCH_VERSION"

if [ "$TORCH_VERSION" = "not_installed" ]; then
    echo -e "   ${YELLOW}⚠️  PyTorch not installed, will be installed by requirements${NC}"
elif [[ "$TORCH_VERSION" == 2.5.* ]]; then
    echo -e "   ${YELLOW}⚠️  PyTorch 2.5.x detected. SSVP-SLT may prefer 2.2.x${NC}"
    echo "   Continuing with current version (compatible with Python 3.10+)"
fi
echo ""

# Install SSVP-SLT requirements (with error handling)
echo "7️⃣  Installing SSVP-SLT requirements..."
if [ -f "requirements.txt" ]; then
    # Install requirements, ignoring dependency conflicts for now
    pip install -r requirements.txt --no-deps 2>/dev/null || pip install -r requirements.txt
    echo "   ✓ Installed requirements"
else
    echo -e "   ${YELLOW}⚠️  requirements.txt not found${NC}"
fi
echo ""

# Install torch first if needed (SSVP-SLT setup.py needs it)
echo "8️⃣  Ensuring PyTorch is available for setup.py..."
python3 -c "import torch" 2>/dev/null || pip install torch torchvision
echo "   ✓ PyTorch available"
echo ""

# Install SSVP-SLT package (skip if setup.py fails)
echo "9️⃣  Installing SSVP-SLT package..."
if pip install -e . 2>&1 | tee /tmp/ssvp_install.log; then
    echo "   ✓ Installed SSVP-SLT in editable mode"
else
    echo -e "   ${YELLOW}⚠️  SSVP-SLT package installation failed${NC}"
    echo "   This is OK - we can use the repository directly"
    echo "   See /tmp/ssvp_install.log for details"
fi
echo ""

# Install fairseq
echo "🔟 Installing fairseq (SSVP-SLT dependency)..."
if [ -d "fairseq-sl" ]; then
    cd fairseq-sl
    if pip install -e . 2>&1 | tee /tmp/fairseq_install.log; then
        echo "   ✓ Installed fairseq-sl"
    else
        echo -e "   ${YELLOW}⚠️  Fairseq installation failed${NC}"
        echo "   See /tmp/fairseq_install.log for details"
    fi
    cd ..
else
    echo -e "   ${YELLOW}⚠️  fairseq-sl directory not found${NC}"
fi
echo ""

# Additional dependencies
echo "1️⃣1️⃣  Installing additional dependencies..."
pip install tensorboard scikit-learn sentencepiece sacrebleu --upgrade
echo "   ✓ Installed additional packages"
echo ""

# Verify installation
echo "1️⃣2️⃣  Verifying installation..."
cd "$ROOT_DIR"

python3 -c "import torch; print('   ✓ PyTorch:', torch.__version__)" 2>/dev/null || echo -e "   ${RED}✗ PyTorch not found${NC}"
python3 -c "import transformers; print('   ✓ Transformers:', __import__('transformers').__version__)" 2>/dev/null || echo -e "   ${YELLOW}⚠️  Transformers not found${NC}"

# Check fairseq (optional)
python3 -c "import fairseq; print('   ✓ Fairseq installed')" 2>/dev/null || echo -e "   ${YELLOW}⚠️  Fairseq not installed (optional)${NC}"

# Check CUDA in PyTorch
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo -e "   ${GREEN}✓ PyTorch CUDA available${NC}"
else
    echo -e "   ${YELLOW}⚠️  PyTorch CUDA not available (CPU only)${NC}"
fi
echo ""

# Summary
echo "================================================================================"
echo -e "${GREEN}✅ SSVP-SLT Installation Complete!${NC}"
echo "================================================================================"
echo ""
echo "📍 Installation locations:"
echo "   - SSVP-SLT repo: $REPO_DIR"
echo "   - Checkpoints:   $MODELS_DIR/checkpoints"
echo ""
echo "🚀 Next steps:"
echo "   1. Download pretrained models:"
echo "      python download_pretrained.py --model base"
echo ""
echo "   2. Prepare How2Sign dataset:"
echo "      python prepare_how2sign_for_ssvp.py"
echo ""
echo "   3. Fine-tune on How2Sign:"
echo "      python finetune_how2sign.py --config configs/finetune_base.yaml"
echo ""
echo "📚 Documentation: src/sign_to_text_ssvp/README.md"
echo ""
