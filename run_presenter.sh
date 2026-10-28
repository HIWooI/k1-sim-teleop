#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "$0")" && pwd)
if [[ ! -x "$repo_dir/.venv/bin/python" ]]; then
  echo 'Run: PYTHON=python3.12 bash scripts/setup_presenter.sh' >&2
  exit 1
fi
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
exec "$repo_dir/.venv/bin/python" -m k1live.presenter "$@"
