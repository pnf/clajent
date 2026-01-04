#!/usr/bin/env bash
set -e  # Exit on error

echo "=== Clajent Environment Setup ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Setting up environment in: $SCRIPT_DIR"
echo ""

# Check for required system dependencies
echo "Checking system dependencies..."

# Check for Java
if ! command -v java &> /dev/null; then
    print_error "Java is not installed. Please install Java 11 or higher."
    exit 1
fi
print_status "Java found: $(java -version 2>&1 | head -n 1)"

# Check for Python3
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
print_status "Python3 found: $(python3 --version)"

# Check for curl
if ! command -v curl &> /dev/null; then
    print_error "curl is not installed. Please install curl."
    exit 1
fi
print_status "curl found"

echo ""
echo "=== Installing Leiningen ==="

# Install Leiningen if not present
if ! command -v lein &> /dev/null; then
    echo "Installing Leiningen..."

    # Create local bin directory if it doesn't exist
    mkdir -p "$HOME/.local/bin"

    # Download lein script
    curl -sL https://raw.githubusercontent.com/technomancy/leiningen/stable/bin/lein \
        -o "$HOME/.local/bin/lein"

    chmod +x "$HOME/.local/bin/lein"

    # Add to PATH if not already there
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        export PATH="$HOME/.local/bin:$PATH"
        print_warning "Added $HOME/.local/bin to PATH for this session"
        print_warning "Add this to your ~/.bashrc or ~/.zshrc:"
        echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi

    # Download leiningen jar
    echo "Downloading Leiningen standalone jar..."
    "$HOME/.local/bin/lein" version

    print_status "Leiningen installed to $HOME/.local/bin/lein"
else
    print_status "Leiningen already installed: $(lein version | head -n 1)"
fi

echo ""
echo "=== Installing Babashka ==="

# Install Babashka if not present
if ! command -v bb &> /dev/null; then
    echo "Installing Babashka..."

    # Download install script
    curl -sL https://raw.githubusercontent.com/babashka/babashka/master/install \
        -o /tmp/install-bb.sh

    chmod +x /tmp/install-bb.sh

    # Install to /usr/local/bin if we have sudo, otherwise to ~/.local/bin
    if command -v sudo &> /dev/null && sudo -n true 2>/dev/null; then
        sudo /tmp/install-bb.sh --dir /usr/local/bin
        print_status "Babashka installed to /usr/local/bin"
    else
        /tmp/install-bb.sh --dir "$HOME/.local/bin"
        print_status "Babashka installed to $HOME/.local/bin"

        if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
            export PATH="$HOME/.local/bin:$PATH"
        fi
    fi

    rm /tmp/install-bb.sh
else
    print_status "Babashka already installed: $(bb --version)"
fi

echo ""
echo "=== Setting up Python Virtual Environment ==="

# Create Python virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created at ./venv"
else
    print_status "Virtual environment already exists"
fi

# Activate venv and install dependencies
echo "Installing Python dependencies..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install Python packages
echo "Installing required Python packages..."
pip install --quiet \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    openai \
    mcp \
    pydantic

print_status "Python dependencies installed"

deactivate

echo ""
echo "=== Installing Clojure Dependencies ==="

# Download Clojure dependencies
echo "Downloading Clojure dependencies (this may take a while)..."
if command -v lein &> /dev/null; then
    lein deps
    print_status "Clojure dependencies installed"
else
    print_warning "Leiningen not in PATH, skipping dependency download"
    print_warning "Run 'lein deps' manually after adding lein to PATH"
fi

echo ""
echo "=== Verifying Environment Variables ==="

# Check for OPEN_ROUTER_KEY
if [ -z "$OPEN_ROUTER_KEY" ]; then
    print_warning "OPEN_ROUTER_KEY environment variable is not set"
    echo "    To use the OpenRouter API, set this variable:"
    echo "    export OPEN_ROUTER_KEY='your-api-key-here'"
    echo ""
    echo "    Add it to your shell config (~/.bashrc or ~/.zshrc) to make it permanent"
else
    print_status "OPEN_ROUTER_KEY is set"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Activate the Python virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Set the OPEN_ROUTER_KEY environment variable if not already set:"
echo "     export OPEN_ROUTER_KEY='your-api-key-here'"
echo ""
echo "  3. Test the Clojure REPL:"
echo "     lein repl"
echo ""
echo "  4. Or run the main function:"
echo "     lein run -m clajent.core/go"
echo ""
echo "  5. Test Python interop:"
echo "     python timeseries/timesense/basic_usage.py"
echo ""
print_status "Setup complete!"
