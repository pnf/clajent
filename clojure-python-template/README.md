# Clojure + Python/Pandas Interop Template

A minimal template for Clojure projects with Python interoperability using [libpython-clj](https://github.com/clj-python/libpython-clj), focused on pandas DataFrames and numpy arrays.

## Quick Start

```bash
# 1. Run setup
./setup.sh

# 2. Start REPL
lein repl

# 3. In REPL, run demo
(require '[example.core :as c])
(c/demo)
```

## What's Included

| File | Purpose |
|------|---------|
| `setup.sh` | Installs Leiningen, Babashka, clj-nrepl-eval, Python venv |
| `clean.sh` | Removes all installed artifacts for clean testing |
| `project.clj` | Clojure dependencies (libpython-clj) |
| `python.edn` | Python interpreter configuration |
| `CLAUDE.md` | Comprehensive guide for AI assistants |
| `src/example/core.clj` | Example pandas/numpy interop code |
| `test/example/verify_test.clj` | Installation verification tests |

## Prerequisites

- Java 11+
- Python 3.8+
- curl

## Verification

```bash
# Run tests to verify installation
lein test
```

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed usage instructions, including:
- REPL workflow with clj-nrepl-eval
- Python interop syntax (py., py.., py.-)
- pandas DataFrame operations
- numpy array operations
- Troubleshooting guide

## License

EPL-2.0 OR GPL-2.0-or-later
