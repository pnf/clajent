# Clajent Environment Setup Guide

This guide walks through setting up the development environment for Clajent, a Clojure project with Python interoperability.

## Prerequisites

Before running the setup script, ensure you have:

- **Java** 11 or higher ([Download](https://adoptium.net/))
- **Python** 3.8 or higher ([Download](https://www.python.org/downloads/))
- **curl** (usually pre-installed on Unix systems)
- **git** (for cloning the repository)

## Quick Setup

Run the automated setup script:

```bash
./setup.sh
```

This will:
1. Install Leiningen (Clojure build tool)
2. Install Babashka (Clojure scripting interpreter)
3. Create a Python virtual environment
4. Install all Python dependencies
5. Download all Clojure dependencies

## Environment Variables

### Required

**`OPEN_ROUTER_KEY`** - Your OpenRouter API key for accessing LLM services

```bash
export OPEN_ROUTER_KEY='sk-or-v1-your-key-here'
```

Add this to your shell configuration file (`~/.bashrc`, `~/.zshrc`, etc.) to make it permanent:

```bash
echo 'export OPEN_ROUTER_KEY="sk-or-v1-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## Manual Setup (if automated script fails)

### 1. Install Leiningen

```bash
mkdir -p ~/.local/bin
curl -sL https://raw.githubusercontent.com/technomancy/leiningen/stable/bin/lein \
    -o ~/.local/bin/lein
chmod +x ~/.local/bin/lein
export PATH="$HOME/.local/bin:$PATH"
lein version  # This downloads the standalone jar
```

### 2. Install Babashka

```bash
curl -sL https://raw.githubusercontent.com/babashka/babashka/master/install \
    -o /tmp/install-bb.sh
chmod +x /tmp/install-bb.sh
/tmp/install-bb.sh --dir ~/.local/bin
```

### 3. Setup Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn matplotlib openai mcp pydantic
```

### 4. Install Clojure Dependencies

```bash
lein deps
```

## Verifying the Setup

### Test Python Environment

```bash
source venv/bin/activate
python timeseries/timesense/basic_usage.py
```

### Test Clojure REPL

```bash
lein repl
```

In the REPL:
```clojure
(require '[clajent.core :as core])
(core/go)
```

### Test Environment Variable Access

**Clojure:**
```bash
bb -e "(println \"OPEN_ROUTER_KEY:\" (System/getenv \"OPEN_ROUTER_KEY\"))"
```

**Python:**
```bash
python -c "import os; print('OPEN_ROUTER_KEY:', os.getenv('OPEN_ROUTER_KEY'))"
```

## Project Structure

```
clajent/
├── setup.sh              # Automated setup script
├── SETUP.md             # This file
├── project.clj          # Leiningen project configuration
├── python.edn           # Python executable configuration for libpython-clj
├── venv/                # Python virtual environment (created by setup)
├── src/
│   └── clajent/
│       ├── core.clj     # Main Clojure namespace
│       └── ...
└── timeseries/
    └── timesense/
        ├── basic_usage.py
        ├── ts_server.py
        └── ...
```

## Configuration Files

### `python.edn`

Configures the Python executable path for libpython-clj:

```clojure
{:python-executable "/path/to/project/venv/bin/python"
 :python-verbose true}
```

This file is automatically configured to use the local virtual environment.

## Troubleshooting

### "Command not found: lein" or "Command not found: bb"

Add `~/.local/bin` to your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Make it permanent by adding to your shell config:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### "ModuleNotFoundError: No module named 'matplotlib'"

Ensure the virtual environment is activated:

```bash
source venv/bin/activate
pip install matplotlib
```

### "Failed to find python executable"

The `python.edn` file needs to point to the correct Python executable. Update it:

```clojure
{:python-executable "/absolute/path/to/venv/bin/python"
 :python-verbose true}
```

### "OPEN_ROUTER_KEY environment variable is not set"

Set the environment variable before running the code:

```bash
export OPEN_ROUTER_KEY='your-api-key-here'
```

### Java version issues

Ensure you have Java 11 or higher:

```bash
java -version
```

If you have multiple Java versions, set `JAVA_HOME`:

```bash
export JAVA_HOME=/path/to/java11
```

## Development Workflow

1. **Activate Python environment** (for Python work):
   ```bash
   source venv/bin/activate
   ```

2. **Start Clojure REPL** (for interactive development):
   ```bash
   lein repl
   ```

3. **Run main function**:
   ```bash
   lein run -m clajent.core/go
   ```

4. **Run Python scripts**:
   ```bash
   python timeseries/timesense/basic_usage.py
   ```

5. **Run tests** (if available):
   ```bash
   lein test
   ```

## Additional Resources

- [Leiningen Documentation](https://leiningen.org/)
- [Babashka Documentation](https://book.babashka.org/)
- [libpython-clj Documentation](https://github.com/clj-python/libpython-clj)
- [OpenRouter API Documentation](https://openrouter.ai/docs)

## Support

If you encounter issues not covered in this guide, please check:
1. The project's GitHub issues
2. The error messages in `/tmp/clojure-*.edn` files
3. Python error tracebacks for dependency issues
