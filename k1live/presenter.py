"""Single-PC presenter launcher: camera/video -> MediaPipe -> standing K1 arms."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request
import webbrowser

REPO = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', default='0', help='camera index or video path')
    ap.add_argument('--scene', choices=('snu', 'plain'), default='snu')
    ap.add_argument('--crop', help='person region x0,y0,x1,y1 (0..1)')
    ap.add_argument('--port-offset', type=int, default=int(os.environ.get('K1LIVE_PORT_OFFSET', '0')))
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--max-seconds', type=float, default=0, help='0 = run until Ctrl+C')
    ap.add_argument('--session-dir', type=Path)
    args = ap.parse_args(argv)
    if not 0 <= args.port_offset <= 56833:
        ap.error('--port-offset must be between 0 and 56833')
    if not 0 <= args.max_seconds < float('inf'):
        ap.error('--max-seconds must be finite and >= 0')
    return args


def open_when_ready(url):
    for _ in range(100):
        try:
            with urllib.request.urlopen(url, timeout=0.5):
                webbrowser.open(url + 'display')
                return
        except OSError:
            time.sleep(0.2)


def main(argv=None):
    args = parse_args(argv)
    # Resolve user paths before changing to the repository for child process imports.
    source = args.source if args.source.isdigit() else str(Path(args.source).expanduser().resolve())
    session = args.session_dir.expanduser().resolve() if args.session_dir else None
    os.environ['K1LIVE_PORT_OFFSET'] = str(args.port_offset)
    os.environ['K1LIVE_RENDER_SCENE'] = args.scene
    if sys.platform.startswith('linux'):
        os.environ.setdefault('MUJOCO_GL', 'egl')
    os.chdir(REPO)
    # Port constants must be imported after setting the offset (also inherited by children).
    from . import service

    service.SETTINGS_PATH = service.LOG_ROOT / 'presenter-settings.json'
    cmd = ['--config', str(REPO / 'k1live/web/presenter.yaml'), '--no-saved-settings',
           '--source', source, '--start', '--max-seconds', str(args.max_seconds)]
    if args.crop:
        cmd += ['--crop', args.crop]
    if session:
        cmd += ['--session-dir', str(session)]
    if not args.no_browser:
        threading.Thread(target=open_when_ready,
                         args=(f'http://127.0.0.1:{8700 + args.port_offset}/',), daemon=True).start()
    return service.main(cmd)


if __name__ == '__main__':
    raise SystemExit(main())
