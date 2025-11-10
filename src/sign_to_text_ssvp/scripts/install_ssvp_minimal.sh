#!/bin/bash
################################################################################
# SSVP-SLT Minimal Installation Script
################################################################################
# Versione semplificata che evita conflitti di dipendenze
# Clona il repository e installa solo le dipendenze essenziali
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "🚀 SSVP-SLT Minimal Installation"
echo "================================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$ROOT_DIR/models"
REPO_DIR="$MODELS_DIR/ssvp_slt_repo"

echo "📂 Setup directory: $REPO_DIR"
echo ""

# Create directories
echo "1️⃣  Creating directories..."
mkdir -p "$MODELS_DIR/checkpoints"
echo "   ✓ Created: $MODELS_DIR/checkpoints"
echo ""

# Clone repository
echo "2️⃣  Cloning SSVP-SLT repository..."
if [ -d "$REPO_DIR" ]; then
    echo -e "   ${YELLOW}⚠️  Repository already exists${NC}"
else
    git clone https://github.com/facebookresearch/ssvp_slt.git "$REPO_DIR"
    echo "   ✓ Cloned SSVP-SLT"
fi
echo ""

# Install minimal dependencies
echo "3️⃣  Installing minimal dependencies..."
pip install --upgrade \
    tensorboard \
    sentencepiece \
    sacrebleu \
    opencv-python \
    ffmpeg-python \
    pyyaml \
    tqdm
echo "   ✓ Installed core dependencies"
echo ""

# Verify
echo "4️⃣  Verifying installation..."
python3 -c "import torch; print('   ✓ PyTorch:', torch.__version__)"
python3 -c "import transformers; print('   ✓ Transformers:', __import__('transformers').__version__)"
echo ""

# Summary
echo "================================================================================"
echo -e "${GREEN}✅ SSVP-SLT Minimal Setup Complete!${NC}"
echo "================================================================================"
echo ""
echo "📍 Repository cloned to: $REPO_DIR"
echo ""
echo "⚠️  Note: This is a minimal setup."
echo "   The SSVP-SLT package itself is NOT installed (due to dependency conflicts)."
echo "   You can use the repository directly for reference and implementation."
echo ""
echo "🚀 Next steps:"
echo "   1. Study SSVP-SLT code:"
echo "      cd $REPO_DIR"
echo "      cat translation/README.md"
echo ""
echo "   2. Download pretrained models:"
echo "      cd $ROOT_DIR"
echo "      python download_pretrained.py --model base"
echo ""
echo "   3. Prepare dataset:"
echo "      bash scripts/prepare_all_splits.sh"
echo ""
echo "   4. Implement fine-tuning script using SSVP-SLT as reference"
echo ""
