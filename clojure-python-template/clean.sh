#!/usr/bin/env bash
set -e

echo "=== Cleaning Clojure + Python Environment ==="
echo ""
echo "This script removes all local installations, virtual environments,"
echo "and cached files to simulate a fresh VM for testing setup.sh."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[REMOVED]${NC} $1"
}

print_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
}

# Get script directory (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ============================================================================
# Kill any running nREPL processes
# ============================================================================

echo "=== Stopping Running Processes ==="

# Kill any lein repl processes
if pgrep -f "lein.*repl" > /dev/null 2>&1; then
    pkill -f "lein.*repl" || true
    print_status "Killed lein repl processes"
else
    print_skip "No lein repl processes found"
fi

# Kill any Java processes that might be nREPL servers
if pgrep -f "clojure.*nrepl" > /dev/null 2>&1; then
    pkill -f "clojure.*nrepl" || true
    print_status "Killed nREPL server processes"
else
    print_skip "No nREPL server processes found"
fi

# ============================================================================
# Remove Project-Level Files
# ============================================================================

echo ""
echo "=== Removing Project-Level Files ==="

# Remove Python virtual environment
if [ -d "venv" ]; then
    rm -rf venv
    print_status "venv/ directory"
else
    print_skip "venv/ directory (not found)"
fi

# Remove Leiningen target directory
if [ -d "target" ]; then
    rm -rf target
    print_status "target/ directory"
else
    print_skip "target/ directory (not found)"
fi

# Remove .nrepl-port file
if [ -f ".nrepl-port" ]; then
    rm -f .nrepl-port
    print_status ".nrepl-port file"
else
    print_skip ".nrepl-port file (not found)"
fi

# Remove Python cache
if [ -d "__pycache__" ]; then
    rm -rf __pycache__
    print_status "__pycache__/ directory"
else
    print_skip "__pycache__/ directory (not found)"
fi

# Remove .clj-kondo cache
if [ -d ".clj-kondo" ]; then
    rm -rf .clj-kondo
    print_status ".clj-kondo/ directory"
else
    print_skip ".clj-kondo/ directory (not found)"
fi

# ============================================================================
# Remove Global Tools (Optional - controlled by flags)
# ============================================================================

REMOVE_GLOBAL=false
if [[ "$1" == "--full" || "$1" == "-f" ]]; then
    REMOVE_GLOBAL=true
fi

if [ "$REMOVE_GLOBAL" = true ]; then
    echo ""
    echo "=== Removing Global Tools (--full mode) ==="

    # Remove Leiningen
    if [ -f "$HOME/.local/bin/lein" ]; then
        rm -f "$HOME/.local/bin/lein"
        print_status "\$HOME/.local/bin/lein"
    fi

    # Remove Leiningen data directory
    if [ -d "$HOME/.lein" ]; then
        rm -rf "$HOME/.lein"
        print_status "\$HOME/.lein/ directory"
    fi

    # Remove Babashka (only if in ~/.local/bin)
    if [ -f "$HOME/.local/bin/bb" ]; then
        rm -f "$HOME/.local/bin/bb"
        print_status "\$HOME/.local/bin/bb"
    fi

    # Remove clj-nrepl-eval
    if [ -f "$HOME/.local/bin/clj-nrepl-eval" ]; then
        rm -f "$HOME/.local/bin/clj-nrepl-eval"
        print_status "\$HOME/.local/bin/clj-nrepl-eval"
    fi

    # Remove Maven repository (Clojure dependencies)
    if [ -d "$HOME/.m2/repository" ]; then
        rm -rf "$HOME/.m2/repository"
        print_status "\$HOME/.m2/repository/ directory"
    fi
else
    echo ""
    echo "=== Skipping Global Tools ==="
    echo "  Use --full or -f flag to also remove:"
    echo "    - \$HOME/.local/bin/lein"
    echo "    - \$HOME/.local/bin/bb"
    echo "    - \$HOME/.local/bin/clj-nrepl-eval"
    echo "    - \$HOME/.lein/"
    echo "    - \$HOME/.m2/repository/"
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=== Clean Complete ==="
echo ""
echo "Project is now in a clean state."
if [ "$REMOVE_GLOBAL" = true ]; then
    echo "Global tools have also been removed."
fi
echo ""
echo "Run ./setup.sh to reinstall everything."
