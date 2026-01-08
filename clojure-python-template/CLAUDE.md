# Clojure + Python/Pandas Interop Environment

This is a template project for Clojure development with Python interoperability, specifically for working with pandas DataFrames and numpy arrays.

## Quick Setup

```bash
./setup.sh
```

This installs: Leiningen, Babashka, clj-nrepl-eval, Python venv with numpy/pandas/matplotlib.

## Prerequisites

- **Java 11+** - Check with `java -version`
- **Python 3.8+** - Check with `python3 --version`
- **curl** - For downloading tools

## REPL Workflow

### Starting the REPL

```bash
lein repl
```

This starts an nREPL server. Note the port number displayed (e.g., `nREPL server started on port 7888`).

### Using clj-nrepl-eval (from another terminal)

```bash
# Discover running REPLs
clj-nrepl-eval --discover-ports

# Evaluate code
clj-nrepl-eval -p <port> "(+ 1 2 3)"

# With timeout (milliseconds)
clj-nrepl-eval -p <port> --timeout 5000 "<code>"
```

**Important:** Always use `:reload` when requiring namespaces to pick up file changes:

```bash
clj-nrepl-eval -p <port> "(require '[example.core :as c] :reload)"
```

## Python Interop Basics

### Initialization

Python must be initialized before any interop calls:

```clojure
(require '[libpython-clj2.python :refer [py. py.. py.-] :as py])
(py/initialize!)
```

### Importing Python Modules

```clojure
(require '[libpython-clj2.require :refer [require-python]])

(require-python '[numpy :as np])
(require-python '[pandas :as pd])
(require-python '[matplotlib.pyplot :as plt])
```

### Calling Python Methods

| Syntax | Use Case | Example |
|--------|----------|---------|
| `(py. obj method args...)` | Call method on object | `(py. df head 5)` |
| `(py.. obj m1 m2 ...)` | Chain method calls | `(py.. df head to_dict)` |
| `(py.- obj attr)` | Access attribute | `(py.- df shape)` |

### Working with Pandas DataFrames

```clojure
;; Create DataFrame from Clojure map
(def df (pd/DataFrame {"a" [1 2 3] "b" [4 5 6]}))

;; Access shape
(py.- df shape)  ; => (3, 2)

;; Get column
(py. df __getitem__ "a")

;; Filter rows
(py. df query "a > 1")

;; Convert to Clojure
(let [records (py. df to_dict :orient "records")]
  (mapv #(into {} %) records))
```

### Working with Numpy Arrays

```clojure
;; Create array
(def arr (np/array [1 2 3 4 5]))

;; Operations
(np/sum arr)
(np/mean arr)
(np/std arr)

;; Convert to Clojure
(vec arr)

;; Convert float result to Clojure number
(py. (np/sum arr) __float__)
```

## Running Tests

```bash
# Run all tests
lein test

# In REPL
(require '[example.verify-test :as t] :reload)
(t/run-all-tests)
```

The test suite verifies:
- Python initialization
- numpy import and basic operations
- pandas DataFrame creation and operations
- pandas Series operations
- matplotlib import

## Project Structure

```
.
├── setup.sh           # Installs all dependencies
├── clean.sh           # Removes local installations for fresh testing
├── project.clj        # Leiningen config (dependencies)
├── python.edn         # Python interpreter config (points to ./venv/bin/python)
├── CLAUDE.md          # This file
├── src/example/
│   └── core.clj       # Example pandas/numpy interop code
├── test/example/
│   └── verify_test.clj # Installation verification tests
└── .claude/
    ├── settings.local.json    # Tool permissions
    └── skills/clojure-eval/   # clj-nrepl-eval skill docs
```

## Configuration Files

### python.edn

Controls libpython-clj's Python interpreter selection:

```clojure
{:python-executable "./venv/bin/python"
 :python-verbose true}
```

Uses relative path for portability across systems.

### project.clj

Key dependencies:
- `org.clojure/clojure "1.12.0"` - Clojure runtime
- `clj-python/libpython-clj "2.026"` - Python interop bridge

## Troubleshooting

### "Could not find python executable"

Run `./setup.sh` to create the venv and regenerate `python.edn`.

### "ModuleNotFoundError: No module named 'pandas'"

Ensure venv is properly set up:

```bash
source venv/bin/activate
pip list | grep pandas
deactivate
```

If missing, re-run `./setup.sh`.

### "Command not found: lein"

Add `~/.local/bin` to your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Changes not reflecting in REPL

Always use `:reload` when requiring:

```clojure
(require '[example.core :as c] :reload)
```

## Clean Install Testing

To verify setup works from scratch:

```bash
# Remove local artifacts (keeps global tools)
./clean.sh

# Or remove everything including lein, bb, etc.
./clean.sh --full

# Then re-setup
./setup.sh

# Verify
lein test
```

## Common Patterns

### DataFrame from CSV

```clojure
(def df (pd/read_csv "data.csv"))
```

### Save DataFrame

```clojure
(py. df to_csv "output.csv" :index false)
```

### Group-by Operations

```clojure
(py.. df (groupby "category") sum)
```

### Date Handling

```clojure
(require-python '[datetime :as dt])
(require-python '[pandas :as pd])

(pd/to_datetime "2024-01-15")
(pd/date_range :start "2024-01-01" :periods 10 :freq "D")
```
