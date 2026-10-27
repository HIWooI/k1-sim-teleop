"""Download GEAR-SONIC deployment ONNX models + sample data from Hugging Face.

Mirrors the logic of vendor/GR00T-WholeBodyControl/download_from_hf.py but stores
everything under sonic_sim/models/ (gitignored).
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = "nvidia/GEAR-SONIC"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models"

VARIANTS = {
    "release": ["model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml"],
    "low_latency": [
        "low_latency/model_encoder.onnx",
        "low_latency/model_decoder.onnx",
        "low_latency/observation_config.yaml",
    ],
    "sonic_v1_1": [
        "sonic_v1_1/model_encoder.onnx",
        "sonic_v1_1/model_decoder.onnx",
        "sonic_v1_1/observation_config.yaml",
    ],
}


def _place(src, dst):
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size == Path(src).stat().st_size:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.chmod(0o644)
    shutil.copyfile(src, dst)
    dst.chmod(0o644)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["release", "low_latency"])
    ap.add_argument("--sample", action="store_true", help="also fetch sample_data/*")
    args = ap.parse_args()
    cfg = hf_hub_download(REPO_ID, "config.json")
    _place(cfg, OUT / "config.json")
    for v in args.variants:
        for f in VARIANTS[v]:
            print("download", f, flush=True)
            p = hf_hub_download(REPO_ID, f)
            dst = OUT / v / Path(f).name
            _place(p, dst)
            print(" ->", dst, flush=True)
    if args.sample:
        snapshot_download(REPO_ID, allow_patterns="sample_data/*", local_dir=str(OUT))
        print("sample ->", OUT / "sample_data")


if __name__ == "__main__":
    main()
