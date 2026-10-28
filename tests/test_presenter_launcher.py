"""Portable profile must not inherit an old GPU/full-body configuration."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from k1live import presenter, service
from k1live.sender import parse_args as sender_args


def test_profile_ignores_saved_backend_and_does_not_require_smpl(tmp_path, monkeypatch):
    saved = tmp_path / 'settings.json'
    saved.write_text(json.dumps(dict(backend='sonic_human', perception='gemx', walk=True)))
    monkeypatch.setattr(service, 'SETTINGS_PATH', saved)
    profile = presenter.REPO / 'k1live/web/presenter.yaml'
    cfg = service.load_config(profile, read_saved=False)
    assert (cfg['backend'], cfg['perception'], cfg['walk']) == ('split', 'mediapipe', False)
    assert '--no-smpl' in cfg['sender_cmd'] and '--no-loco' in cfg['sender_cmd']
    assert '--no-walk' in cfg['sim_cmd']
    assert service.load_config(profile)['backend'] == 'sonic_human'


@pytest.mark.parametrize('flag,value', [('--arm-filter-cutoff','0'), ('--arm-filter-cutoff','nan'),
    ('--arm-filter-beta','21'), ('--arm-filter-beta','-1')])
def test_reject_bad_filters(flag, value):
    with pytest.raises(SystemExit):
        sender_args([flag,value])


@pytest.mark.parametrize('args', [['--port-offset','56834'], ['--port-offset','-1'],
                                ['--max-seconds','nan'], ['--max-seconds','-1']])
def test_launcher_rejects_invalid_ports_and_durations(args):
    with pytest.raises(SystemExit):
        presenter.parse_args(args)


def test_latest_pair_has_no_backlog_and_cleans_blob_urls():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required for the display scheduling test')
    subprocess.run([node, 'tests/display-pair.cjs'], cwd=presenter.REPO, check=True,
                   timeout=10, capture_output=True, text=True)
