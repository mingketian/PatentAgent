#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "patent-agent" ]]; then
  echo "Activate the patent-agent conda environment first:"
  echo "  conda activate patent-agent"
  exit 1
fi

cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
python -m streamlit run app.py "$@"
