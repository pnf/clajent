# Clojure Development Tools - Installation Status

**Date:** 2026-01-03
**Environment:** Claude Code Development Container

## Summary

Core Clojure development tools have been installed, but `clj-nrepl-eval` requires network access to download dependencies on first use.

---

## ✅ Successfully Installed

### 1. Leiningen 2.12.0
- **Location:** `/usr/local/bin/lein`
- **Status:** Fully installed and functional
- **Test:** `lein version`
- **Use:** Primary build tool for this Clojure project

### 2. Babashka v1.12.213
- **Location:** `/usr/local/bin/bb`
- **Status:** Fully installed and functional
- **Test:** `bb --version`
- **Use:** Fast Clojure scripting and task runner

### 3. Java OpenJDK 21.0.9
- **Location:** `/usr/bin/java`
- **Status:** Pre-installed and functional
- **Test:** `java -version`

---

## ⚠️ Partially Installed

### clj-nrepl-eval
- **Location:** `~/.local/bin/clj-nrepl-eval` (wrapper script)
- **Source:** `~/.local/share/clojure-mcp-light` (cloned from GitHub v0.2.1)
- **Status:** Script installed but requires network access on first run
- **Issue:** Cannot download Maven dependencies due to network restrictions
- **Workaround:** Will work when network access is available

**First-run requirements:**
- Network access to Maven Central (repo1.maven.org)
- No proxy authentication restrictions

---

## Environment Persistence

**Current Installation Locations:**
```
/usr/local/bin/lein        # System-wide
/usr/local/bin/bb          # System-wide
~/.local/bin/clj-nrepl-eval  # User-level
~/.local/share/clojure-mcp-light/  # User-level source
```

**Persistence depends on:**
1. Whether `/usr/local` is on a persistent volume
2. Whether `$HOME` (`/root`) is on a persistent volume
3. Container/VM configuration

**To check persistence:**
- After a restart, run: `which lein bb clj-nrepl-eval`
- If tools are missing, run: `./setup-clojure-tools.sh`

---

## Quick Start

### Using Leiningen (Works Now)
```bash
cd /home/user/clajent
lein repl  # Start a REPL
lein test  # Run tests
lein run   # Run the project
```

### Using clj-nrepl-eval (Requires Network)
```bash
# Discover running nREPL servers
clj-nrepl-eval --discover-ports

# Connect and evaluate (after REPL is running)
clj-nrepl-eval -p PORT "(+ 1 2 3)"
```

---

## Re-installation

If tools are missing after environment reset:

```bash
cd /home/user/clajent
./setup-clojure-tools.sh
```

This script:
1. Installs Leiningen
2. Installs Babashka
3. Clones clojure-mcp-light
4. Creates wrapper scripts for clj-nrepl-eval

---

## Network Issues Encountered

During installation, the following network issues were encountered:

1. **Proxy authentication errors** - Authenticated proxy blocking downloads
2. **DNS resolution failures** - Unable to reach repo1.maven.org
3. **Maven Central access** - Required for Clojure dependencies

These issues prevent `clj-nrepl-eval` from downloading its runtime dependencies on first use. The tool will work once network access is available.

---

## PATH Configuration

Ensure `~/.local/bin` is in your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Add to `~/.bashrc` for persistence:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

---

## Testing Installation

```bash
# Test Leiningen (should work)
lein version

# Test Babashka (should work)
bb --version

# Test clj-nrepl-eval (requires network)
clj-nrepl-eval --help
```

---

## Additional Tools Available

The following Python development tools are also installed in `~/.local/bin`:
- black, mypy, flake8, pytest, ruff, pyright, poetry

These tools integrate with the Python interop features in this Clojure project (`libpython-clj`).
