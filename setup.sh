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

# ============================================================================
# Check System Dependencies
# ============================================================================

echo "=== Checking System Dependencies ==="

if ! command -v java &> /dev/null; then
    print_error "Java is not installed. Please install Java 11 or higher."
    exit 1
fi
print_status "Java found: $(java -version 2>&1 | head -n 1)"

if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
print_status "Python3 found: $(python3 --version)"

if ! command -v curl &> /dev/null; then
    print_error "curl is not installed. Please install curl."
    exit 1
fi
print_status "curl found"

# ============================================================================
# Install Leiningen
# ============================================================================

echo ""
echo "=== Installing Leiningen ==="

if ! command -v lein &> /dev/null; then
    echo "Installing Leiningen..."
    mkdir -p "$HOME/.local/bin"

    curl -sL https://raw.githubusercontent.com/technomancy/leiningen/stable/bin/lein \
        -o "$HOME/.local/bin/lein"
    chmod +x "$HOME/.local/bin/lein"

    # Add to PATH if not already there
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Download leiningen jar
    echo "Downloading Leiningen standalone jar..."
    "$HOME/.local/bin/lein" version > /dev/null 2>&1

    print_status "Leiningen installed to $HOME/.local/bin/lein"
else
    print_status "Leiningen already installed: $(lein version | head -n 1)"
fi

# ============================================================================
# Install Babashka
# ============================================================================

echo ""
echo "=== Installing Babashka ==="

if ! command -v bb &> /dev/null; then
    echo "Installing Babashka..."

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

# ============================================================================
# Install clj-nrepl-eval
# ============================================================================

echo ""
echo "=== Installing clj-nrepl-eval ==="

if ! command -v clj-nrepl-eval &> /dev/null; then
    echo "Installing clj-nrepl-eval..."

    # Create the clj-nrepl-eval script
    cat > "$HOME/.local/bin/clj-nrepl-eval" << 'NREPL_EOF'
#!/usr/bin/env bb

(require '[clojure.string :as str])
(require '[bencode.core :as bencode])
(require '[clojure.java.io :as io])

(defn print-help []
  (println "clj-nrepl-eval - Evaluate Clojure code via nREPL")
  (println "\nUsage: clj-nrepl-eval [options] \"<code>\"")
  (println "\nOptions:")
  (println "  -p, --port PORT              nREPL port (required)")
  (println "  -H, --host HOST              nREPL host (default: 127.0.0.1)")
  (println "  -t, --timeout MILLISECONDS   Timeout in milliseconds (default: 120000)")
  (println "  -d, --discover-ports         Discover nREPL servers in current directory")
  (println "  -h, --help                   Show this help message"))

(defn discover-nrepl-ports []
  (let [cwd (System/getProperty "user.dir")
        port-files (file-seq (io/file cwd))]
    (println "Discovering nREPL servers in current directory...")
    (doseq [f port-files]
      (when (and (.isFile f)
                 (or (.endsWith (.getName f) ".nrepl-port")
                     (.endsWith (.getName f) ".port")))
        (try
          (let [port (str/trim (slurp f))]
            (println (format "Found nREPL server on port %s (%s)" port (.getName f))))
          (catch Exception e
            nil))))))

(defn connect-and-eval [host port code timeout]
  (try
    (with-open [socket (java.net.Socket. host port)
                in (io/reader socket)
                out (io/writer socket)]
      (let [session-id (str (java.util.UUID/randomUUID))
            msg {:op "eval" :code code :session session-id}
            _ (bencode/write-bencode out msg)
            _ (.flush out)
            start-time (System/currentTimeMillis)]
        (loop []
          (when (> (- (System/currentTimeMillis) start-time) timeout)
            (throw (ex-info "Evaluation timeout" {:timeout timeout})))
          (let [response (bencode/read-bencode in)]
            (when-let [out-val (:out response)]
              (print out-val)
              (flush))
            (when-let [err-val (:err response)]
              (binding [*out* *err*]
                (print err-val)
                (flush)))
            (when-let [value (:value response)]
              (println value))
            (when-not (= "done" (:status response))
              (recur))))))
    (catch java.net.ConnectException e
      (println "Error: Could not connect to nREPL server on" (str host ":" port))
      (System/exit 1))
    (catch Exception e
      (println "Error:" (.getMessage e))
      (System/exit 1))))

(defn parse-args [args]
  (loop [args args
         opts {:host "127.0.0.1" :timeout 120000}
         code-parts []]
    (if (empty? args)
      [opts (str/join " " code-parts)]
      (let [arg (first args)
            rest-args (rest args)]
        (cond
          (or (= arg "-h") (= arg "--help"))
          [:help nil]

          (or (= arg "-d") (= arg "--discover-ports"))
          [:discover nil]

          (or (= arg "-p") (= arg "--port"))
          (if (empty? rest-args)
            (do (println "Error: --port requires a value")
                (System/exit 1))
            (recur (rest rest-args)
                   (assoc opts :port (Integer/parseInt (first rest-args)))
                   code-parts))

          (or (= arg "-H") (= arg "--host"))
          (if (empty? rest-args)
            (do (println "Error: --host requires a value")
                (System/exit 1))
            (recur (rest rest-args)
                   (assoc opts :host (first rest-args))
                   code-parts))

          (or (= arg "-t") (= arg "--timeout"))
          (if (empty? rest-args)
            (do (println "Error: --timeout requires a value")
                (System/exit 1))
            (recur (rest rest-args)
                   (assoc opts :timeout (Integer/parseInt (first rest-args)))
                   code-parts))

          :else
          (recur rest-args opts (conj code-parts arg)))))))

(let [[opts code] (parse-args *command-line-args*)]
  (cond
    (= opts :help)
    (do (print-help) (System/exit 0))

    (= opts :discover)
    (do (discover-nrepl-ports) (System/exit 0))

    (nil? (:port opts))
    (do (println "Error: --port is required")
        (print-help)
        (System/exit 1))

    :else
    (let [code-to-eval (if (str/blank? code) (slurp *in*) code)]
      (when (str/blank? code-to-eval)
        (println "Error: No code provided")
        (System/exit 1))
      (connect-and-eval (:host opts) (:port opts) code-to-eval (:timeout opts)))))
NREPL_EOF

    chmod +x "$HOME/.local/bin/clj-nrepl-eval"

    print_status "clj-nrepl-eval installed to $HOME/.local/bin/clj-nrepl-eval"
else
    print_status "clj-nrepl-eval already installed"
fi

# ============================================================================
# Setup Python Virtual Environment
# ============================================================================

echo ""
echo "=== Setting up Python Virtual Environment ==="

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

pip install --upgrade pip --quiet

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

# ============================================================================
# Configure python.edn with correct venv path
# ============================================================================

echo ""
echo "=== Configuring python.edn ==="

cat > python.edn << EOF
{:python-executable "./venv/bin/python"
 :python-verbose true
 }
EOF

print_status "python.edn configured with venv path: ./venv/bin/python"

# ============================================================================
# Install Clojure Dependencies
# ============================================================================

echo ""
echo "=== Installing Clojure Dependencies ==="

if command -v lein &> /dev/null; then
    echo "Downloading Clojure dependencies (this may take a while)..."
    lein deps
    print_status "Clojure dependencies installed"
else
    print_warning "Leiningen not in PATH, skipping dependency download"
fi

# ============================================================================
# Verify Installation
# ============================================================================

echo ""
echo "=== Verifying Installation ==="

# Test Babashka
if bb -e "(println \"Babashka works!\")" 2>&1 | grep -q "Babashka works!"; then
    print_status "Babashka verification passed"
else
    print_warning "Babashka verification failed"
fi

# Test Python
source venv/bin/activate
if python -c "import numpy, matplotlib; print('Python imports work!')" 2>&1 | grep -q "Python imports work!"; then
    print_status "Python verification passed"
else
    print_warning "Python verification failed"
fi
deactivate

# Test clj-nrepl-eval (basic check)
if command -v clj-nrepl-eval &> /dev/null; then
    print_status "clj-nrepl-eval is available"
    echo "  To test: Start a REPL with 'lein repl' then use 'clj-nrepl-eval --discover-ports'"
else
    print_warning "clj-nrepl-eval not found in PATH"
fi

# ============================================================================
# Check Environment Variables
# ============================================================================

echo ""
echo "=== Checking Environment Variables ==="

if [ -z "$OPEN_ROUTER_KEY" ]; then
    print_warning "OPEN_ROUTER_KEY environment variable is not set"
    echo ""
    echo "  Set it with:"
    echo "    export OPEN_ROUTER_KEY='sk-or-v1-your-key-here'"
    echo ""
    echo "  Make it permanent by adding to ~/.bashrc or ~/.zshrc"
else
    print_status "OPEN_ROUTER_KEY is set"
fi

# ============================================================================
# Update PATH in shell config
# ============================================================================

echo ""
echo "=== Updating Shell Configuration ==="

if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    print_warning "$HOME/.local/bin is not in your PATH"
    echo ""
    echo "  Add this to your ~/.bashrc or ~/.zshrc:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

# ============================================================================
# Setup Complete
# ============================================================================

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Tools installed:"
echo "  • Leiningen (lein) - Clojure build tool"
echo "  • Babashka (bb) - Clojure scripting"
echo "  • clj-nrepl-eval - nREPL evaluation tool"
echo "  • Python venv with all dependencies"
echo ""
echo "Next steps:"
echo "  1. Ensure ~/.local/bin is in your PATH"
echo "  2. Set OPEN_ROUTER_KEY environment variable"
echo "  3. See README.md for usage instructions"
echo ""
print_status "Setup complete!"
