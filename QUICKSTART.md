# Clajent Quick Start

## Initial Setup (One Time)

```bash
# 1. Clone the repository
git clone <repo-url>
cd clajent

# 2. Run setup script
./setup.sh

# 3. Set API key
export OPEN_ROUTER_KEY='sk-or-v1-your-key-here'
```

## Daily Development

### Activate Python Environment
```bash
source venv/bin/activate
```

### Run Clojure Code
```bash
# Start REPL
lein repl

# Run main function
lein run -m clajent.core/go

# Run with Babashka (faster)
bb -e "(require '[clajent.core :as c]) (c/go)"
```

### Run Python Code
```bash
# Basic usage example
python timeseries/timesense/basic_usage.py

# MCP Server
python timeseries/timesense/ts_server.py
```

## Verify Setup

```bash
# Check environment variable
echo $OPEN_ROUTER_KEY

# Test Clojure environment
bb -e "(println \"Clojure works!\")"

# Test Python environment
python -c "import numpy, matplotlib; print('Python works!')"

# Test that environment variable is accessible
bb -e "(println (System/getenv \"OPEN_ROUTER_KEY\"))"
```

## Common Commands

| Task | Command |
|------|---------|
| Activate Python venv | `source venv/bin/activate` |
| Start Clojure REPL | `lein repl` |
| Run Clojure main | `lein run -m clajent.core/go` |
| Update dependencies | `lein deps` |
| Clean build | `lein clean` |
| Python dependencies | `pip install -r requirements.txt` |

## Troubleshooting Quick Fixes

| Issue | Fix |
|-------|-----|
| "Command not found: lein" | `export PATH="$HOME/.local/bin:$PATH"` |
| "ModuleNotFoundError" | `source venv/bin/activate && pip install <module>` |
| "OPEN_ROUTER_KEY not set" | `export OPEN_ROUTER_KEY='your-key'` |
| Python path error | Update `python.edn` with correct venv path |

## Project Structure

```
clajent/
├── setup.sh           # Run this first!
├── SETUP.md          # Detailed setup guide
├── QUICKSTART.md     # This file
├── src/clajent/      # Clojure source code
├── timeseries/       # Python source code
└── venv/            # Python virtual environment
```

## Environment Variables

**Required:**
- `OPEN_ROUTER_KEY` - OpenRouter API key

**Optional:**
- `JAVA_HOME` - Java installation path
- `LEIN_HOME` - Leiningen home directory

## Need Help?

1. Check `SETUP.md` for detailed instructions
2. Run `./setup.sh` to verify/fix environment
3. Check error logs in `/tmp/clojure-*.edn`
