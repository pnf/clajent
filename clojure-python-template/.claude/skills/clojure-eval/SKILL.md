---
name: clojure-eval
description: Evaluate Clojure code via nREPL using clj-nrepl-eval. Use this when you need to test code, check if edited files compile, verify function behavior, or interact with a running REPL session.
---

# Clojure REPL Evaluation

## When to Use This Skill

Use this skill when you need to:
- **Verify that edited Clojure files compile and load correctly**
- Test function behavior interactively
- Check the current state of the REPL
- Debug code by evaluating expressions
- Require or load namespaces for testing
- Validate that code changes work before committing

## How It Works

The `clj-nrepl-eval` command evaluates Clojure code against an nREPL server. **Session state persists between evaluations**, so you can require a namespace in one evaluation and use it in subsequent calls.

## Instructions

### 1. Discover nREPL Servers

```bash
clj-nrepl-eval --discover-ports
```

If no servers found, start one with `lein repl`.

### 2. Evaluate Clojure Code

```bash
clj-nrepl-eval -p <PORT> "(+ 1 2 3)"
```

### 3. Require and Test Namespaces

Always use `:reload` to pick up changes:

```bash
clj-nrepl-eval -p <PORT> "(require '[example.core :as c] :reload)"
clj-nrepl-eval -p <PORT> "(c/demo)"
```

### 4. Multiple Expressions

```bash
clj-nrepl-eval -p <PORT> "(def x 10) (* x 2)"
```

### 5. With Timeout

```bash
clj-nrepl-eval -p <PORT> --timeout 5000 "(long-running-fn)"
```

## Available Options

- `-p, --port PORT` - nREPL port (required)
- `-H, --host HOST` - nREPL host (default: 127.0.0.1)
- `-t, --timeout MILLISECONDS` - Timeout (default: 120000)
- `-d, --discover-ports` - Discover nREPL servers
- `-h, --help` - Show help

## Typical Workflow

1. Start REPL: `lein repl`
2. Discover port: `clj-nrepl-eval --discover-ports`
3. Require namespace: `clj-nrepl-eval -p <PORT> "(require '[ns :as n] :reload)"`
4. Test function: `clj-nrepl-eval -p <PORT> "(n/my-fn args)"`
5. Iterate: Make changes, re-require with `:reload`, test again
