#!/usr/bin/env bash
# Setup script for Clojure development tools
# This script installs the necessary tools for Clojure development with Claude Code

set -e

echo "========================================="
echo "Clojure Development Tools Setup"
echo "========================================="
echo ""

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   SUDO=""
else
   SUDO="sudo"
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

#Install Leiningen
echo "[1/4] Installing Leiningen..."
if command_exists lein; then
    echo "  ✓ Leiningen already installed: $(lein version | head -1)"
else
    curl -o /tmp/lein https://raw.githubusercontent.com/technomancy/leiningen/stable/bin/lein
    chmod +x /tmp/lein
    $SUDO mv /tmp/lein /usr/local/bin/lein
    echo "  Running lein for first-time setup..."
    lein version
    echo "  ✓ Leiningen installed successfully"
fi
echo ""

# Install Babashka
echo "[2/4] Installing Babashka..."
if command_exists bb; then
    echo "  ✓ Babashka already installed: $(bb --version)"
else
    echo "  Downloading Babashka..."
    curl -sL https://github.com/babashka/babashka/releases/download/v1.12.213/babashka-1.12.213-linux-amd64.tar.gz -o /tmp/bb.tar.gz
    $SUDO tar -xzf /tmp/bb.tar.gz -C /usr/local/bin
    rm /tmp/bb.tar.gz
    echo "  ✓ Babashka installed: $(bb --version)"
fi
echo ""

# Clone clojure-mcp-light
echo "[3/4] Setting up clojure-mcp-light..."
CLOJURE_MCP_DIR="$HOME/.local/share/clojure-mcp-light"
if [ -d "$CLOJURE_MCP_DIR" ]; then
    echo "  ✓ clojure-mcp-light already exists at $CLOJURE_MCP_DIR"
else
    echo "  Cloning clojure-mcp-light repository..."
    mkdir -p "$HOME/.local/share"
    git clone --depth 1 --branch v0.2.1 https://github.com/bhauman/clojure-mcp-light.git "$CLOJURE_MCP_DIR"
    echo "  ✓ clojure-mcp-light cloned successfully"
fi
echo ""

# Create wrapper scripts
echo "[4/4] Creating wrapper scripts..."
mkdir -p ~/.local/bin

# clj-nrepl-eval
cat > ~/.local/bin/clj-nrepl-eval << 'EOF'
#!/usr/bin/env bash
SCRIPT_DIR="$HOME/.local/share/clojure-mcp-light"
cd "$SCRIPT_DIR" && exec bb -m clojure-mcp-light.nrepl-eval "$@"
EOF
chmod +x ~/.local/bin/clj-nrepl-eval
echo "  ✓ Created ~/.local/bin/clj-nrepl-eval"

# clj-paren-repair
cat > ~/.local/bin/clj-paren-repair << 'EOF'
#!/usr/bin/env bash
SCRIPT_DIR="$HOME/.local/share/clojure-mcp-light"
cd "$SCRIPT_DIR" && exec bb -m clojure-mcp-light.paren-repair "$@"
EOF
chmod +x ~/.local/bin/clj-paren-repair
echo "  ✓ Created ~/.local/bin/clj-paren-repair"

echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "Installed tools:"
echo "  • Leiningen:      $(lein version 2>&1 | head -1 || echo 'ERROR')"
echo "  • Babashka:       $(bb --version || echo 'ERROR')"
echo "  • clj-nrepl-eval: ~/.local/bin/clj-nrepl-eval"
echo "  • clj-paren-repair: ~/.local/bin/clj-paren-repair"
echo ""
echo "Make sure ~/.local/bin is in your PATH:"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
echo ""
echo "Test the installation:"
echo "  clj-nrepl-eval --help"
echo ""
