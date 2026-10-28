#!/usr/bin/env bash
# Only the assets needed by the standing-arm presenter; no private policy files.
set -euo pipefail
cd "$(dirname "$0")/.."
presenter_python=${PYTHON:-python3.12}
if [[ ! -x .venv/bin/python ]]; then
  "$presenter_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
  "$presenter_python" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use a Python 3.12 .venv"'
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-presenter.txt

asset_dir=${AI_SAPIENS_ROOT:-third_party/ai_sapiens}
if [[ ! -f "$asset_dir/ai_sapiens_description/mujoco/k1/k1.xml" ]]; then
  mkdir -p "$(dirname "$asset_dir")"
  git clone --depth 1 https://github.com/ROBOTIS-GIT/ai_sapiens.git "$asset_dir"
fi
bash scripts/fetch_weights.sh

mkdir -p models
.venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
import urllib.request

target = Path('models/pose_landmarker_full.task')
expected = '5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1'
if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
    url = ('https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
           'pose_landmarker_full/float16/1/pose_landmarker_full.task')
    temp = target.with_suffix('.download')
    try:
        urllib.request.urlretrieve(url, temp)
        if hashlib.sha256(temp.read_bytes()).hexdigest() != expected:
            raise RuntimeError('MediaPipe model SHA256 mismatch')
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
print('Presenter setup ready. Run: bash run_presenter.sh')
PY
