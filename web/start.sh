#!/usr/bin/env bash
# ── Patent Prior-Art Agent — Web Demo ──
# Usage:  cd patent-agent/web && ./start.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════╗"
echo "║   Patent Prior-Art Agent — Web Demo      ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ensure web deps
pip install -q fastapi "uvicorn[standard]" python-dotenv 2>/dev/null || true

echo "Starting server on http://localhost:8899"
echo ""
python server.py
