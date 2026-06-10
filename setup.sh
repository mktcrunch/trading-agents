#!/bin/bash
# Setup script for MarketCrunch Trading Agents
# Installs dependencies and prepares local ADK environment

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  MarketCrunch Trading Agents - Setup Script                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# Check Python version
echo ""
echo "1️⃣  Checking Python version..."
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "2️⃣  Creating Python virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo ""
    echo "2️⃣  Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "3️⃣  Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "4️⃣  Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"

# Install requirements (minimal first to avoid PyYAML conflicts)
echo ""
echo "5️⃣  Installing dependencies (minimal)..."
pip install -r requirements-minimal.txt
echo "✓ Minimal dependencies installed"

echo ""
echo "5b️⃣  Installing additional dependencies..."
pip install fastapi uvicorn pydantic databento ta-lib black flake8 pytest-cov pytest-asyncio python-json-logger docker cloudpickle backtrader bt
echo "✓ Additional dependencies installed"

# Install Google Cloud SDK (optional but recommended)
echo ""
echo "6️⃣  Checking for Google Cloud SDK..."
if command -v gcloud &> /dev/null; then
    echo "✓ Google Cloud SDK already installed"
else
    echo "⚠️  Google Cloud SDK not found"
    echo "   For Cloud Run deployment, install: https://cloud.google.com/sdk/docs/install"
fi

# Check for ADK
echo ""
echo "7️⃣  Verifying ADK installation..."
python -c "import adk; print(f'✓ ADK version: {adk.__version__}')" || echo "⚠️  ADK not found - install with: pip install google-adk"

# Create .env if not exists
echo ""
echo "8️⃣  Checking environment configuration..."
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found"
    echo "   Copy from .env.template and fill in your API keys:"
    echo "   cp .env.template .env"
else
    echo "✓ .env file exists"
fi

# Create directories
echo ""
echo "9️⃣  Creating data directories..."
mkdir -p data logs reports
echo "✓ Directories created"

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ✓ Setup Complete!                                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure environment variables:"
echo "   cp .env.template .env"
echo "   # Edit .env with your API keys"
echo ""
echo "2. Test connections:"
echo "   export \$(cat .env | xargs)"
echo "   python test_connections.py"
echo ""
echo "3. Start ADK development:"
echo "   adk web"
echo "   # Opens http://localhost:8000"
echo ""
echo "4. Or run from CLI:"
echo "   adk run"
echo ""
