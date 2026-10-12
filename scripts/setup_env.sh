#!/usr/bin/env bash
# Recreate the Python environment for this project without root privileges.
# Verified in the audited environment (Pop!_OS 22.04, Python 3.10.12).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=python3
SITE=".venv/lib/python3.10/site-packages"

if [ ! -x ".venv/bin/python3" ]; then
    echo "[setup] creating venv (without pip; ensurepip unavailable without root)"
    $PY -m venv --without-pip .venv
    mkdir -p "$SITE"
fi

echo "[setup] installing pinned dependencies into $SITE"
pip3 install --no-input --target "$SITE" \
    "numpy==2.2.6" "scipy==1.15.2" "matplotlib==3.10.1" \
    "scikit-learn==1.6.1" "pandas==2.3.3" "pytest==8.3.5" "PyYAML==6.0.3"

echo "[setup] installing CPU-only PyTorch"
pip3 install --no-input --target "$SITE" \
    --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu"

echo "[setup] verifying imports"
.venv/bin/python3 - <<'EOF'
import numpy, scipy, matplotlib, sklearn, pandas, yaml, torch, pytest
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("matplotlib", matplotlib.__version__)
print("sklearn", sklearn.__version__)
print("pandas", pandas.__version__)
print("torch", torch.__version__)
print("OK")
EOF
echo "[setup] done. Use .venv/bin/python3 for all commands."
