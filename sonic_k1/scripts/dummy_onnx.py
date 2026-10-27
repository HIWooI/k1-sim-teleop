"""Write stand-in SONIC encoder/decoder ONNX files with the input sizes a robot implies (plumbing tests only).

    .venv/bin/python sonic_k1/scripts/dummy_onnx.py --robot k1 --variant release --out /tmp/k1_dummy

The decoder is a zero MatMul (action = 0 -> q_target = default pose, i.e. a pure PD stand), the encoder
outputs a zero token. They let the Mac deploy loop (sonic/) run end to end for K1 before a real K1 SONIC
checkpoint exists: observation sizes, joint orders, gains, MuJoCo model, INIT/band procedure.
The ONNX protobuf is encoded by hand because the `onnx` package is not installed in this venv.
"""
from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _int(field: int, v: int) -> bytes:
    return _key(field, 0) + _varint(v)


def _bytes(field: int, b: bytes) -> bytes:
    return _key(field, 2) + _varint(len(b)) + b


def _str(field: int, s: str) -> bytes:
    return _bytes(field, s.encode())


def _value_info(name: str, dims) -> bytes:
    shape = b"".join(_bytes(1, _int(1, d)) for d in dims)  # TensorShapeProto.dim{dim_value}
    tensor_type = _int(1, 1) + _bytes(2, shape)  # elem_type FLOAT, shape
    return _str(1, name) + _bytes(2, _bytes(1, tensor_type))  # ValueInfoProto{name, type{tensor_type}}


def matmul_model(din: int, dout: int, in_name="obs", out_name="out") -> bytes:
    """ModelProto for out = in @ W, W = zeros(din, dout), opset 17."""
    w = np.zeros((din, dout), np.float32).tobytes()
    init = _int(1, din) + _int(1, dout) + _int(2, 1) + _str(8, "W") + _bytes(9, w)  # TensorProto
    node = _str(1, in_name) + _str(1, "W") + _str(2, out_name) + _str(3, "mm") + _str(4, "MatMul")
    graph = (_bytes(1, node) + _str(2, "zero") + _bytes(5, init)
             + _bytes(11, _value_info(in_name, (1, din))) + _bytes(12, _value_info(out_name, (1, dout))))
    opset = _str(1, "") + _int(2, 17)
    return _int(1, 8) + _str(2, "sonic_k1.dummy") + _bytes(7, graph) + _bytes(8, opset)


def main():
    sys.path.insert(0, str(ROOT))
    from sonic.params import MODELS
    from sonic.robots import get_robot, obs_dims

    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="k1")
    ap.add_argument("--variant", default="release")
    ap.add_argument("--out", required=True, help="models dir to create (<out>/<variant>/...)")
    args = ap.parse_args()
    src = MODELS / args.variant / "observation_config.yaml"
    d = obs_dims(get_robot(args.robot), MODELS / args.variant)
    vdir = Path(args.out) / args.variant
    vdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, vdir / "observation_config.yaml")
    (vdir / "model_encoder.onnx").write_bytes(matmul_model(d["encoder"], d["token"]))
    (vdir / "model_decoder.onnx").write_bytes(matmul_model(d["decoder"], d["action"]))
    print(f"{args.robot} {args.variant}: encoder {d['encoder']}->{d['token']}, decoder {d['decoder']}->{d['action']} in {vdir}")


if __name__ == "__main__":
    main()
