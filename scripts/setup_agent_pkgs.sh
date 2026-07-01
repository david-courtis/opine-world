#!/usr/bin/env bash
set -euo pipefail

PKG="${ARC3_AGENT_PYTHONPATH:-$HOME/.arc_agent_pkgs/py312}"
SYS_PY="${ARC3_SANDBOX_PYTHON:-/usr/bin/python3}"
PKGS=("numpy")   # add more here if needed, e.g. "numpy" "scipy" "pillow" "networkx"

echo "[setup_agent_pkgs] target dir : $PKG"
echo "[setup_agent_pkgs] ABI python : $SYS_PY ($("$SYS_PY" --version 2>&1))"
echo "[setup_agent_pkgs] packages   : ${PKGS[*]}"

rm -rf "$PKG"
mkdir -p "$PKG"
uv pip install --python "$SYS_PY" --target "$PKG" "${PKGS[@]}"

echo "[setup_agent_pkgs] verifying import under $SYS_PY ..."
PYTHONPATH="$PKG" "$SYS_PY" -c "import numpy; print('  numpy', numpy.__version__, 'OK')"
echo "[setup_agent_pkgs] done. sandbox.py binds this dir ro + sets PYTHONPATH automatically."
