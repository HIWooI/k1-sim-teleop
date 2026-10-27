# K1 정책 가중치

행사 설정(hybrid)과 split 백엔드가 쓰는 K1 정책 세 개를 GitHub Release
[`weights-v1`](https://github.com/HIWooI/k1-sim-teleop/releases/tag/weights-v1)로 배포한다. git에는 넣지 않는다(`.gitignore`).

## 받기

```bash
scripts/fetch_weights.sh     # curl·tar만 필요. SHA256SUMS로 확인하고 제자리에 푼다. 이미 있으면 받지 않는다
```

손으로 받을 때는 Release의 `k1-policies.tar.gz`를 저장소 루트에서 `tar -xzf`로 푼다. 파일별 SHA-256은 Release의 `SHA256SUMS`에 있다.

## 들어 있는 것

| 경로 | 정책 | 입력 → 출력 | 쓰는 곳 |
|---|---|---|---|
| `k1/assets/loco_armdriven_r2/exported/policy.onnx` | 보행 정책 | 관측 390 → 관절 23 | split·hybrid(서 있을 때)의 다리 |
| `k1/assets/student_asset/exported/policy.onnx` | mimic student | 관측 124 → 관절 23 | split의 팔, hybrid의 그림자 실행 |
| `models/k1/k1s_walk4/deploy/k1s_walk4_007000/` | K1 SONIC 보행 walk4 7000 (`model_encoder.onnx`, `model_decoder.onnx`, `observation_config.yaml`) | 인코더 1135 → 토큰 64, 디코더 814 → 관절 23 | hybrid가 걸을 때(행사 설정) |

## 만든 방법

- `loco_armdriven_r2`(2026-08-21), `student_asset`(2026-08-14): 저장소 작성자가 ROBOTIS cyclo_lab(Apache-2.0) 학습 환경으로 학습했다.
  배포 파라미터는 같은 폴더의 `params/sim2real.yaml`, 관측·action 변환은 `k1/assets/student_asset/expert_adapter.py`에 있다.
- walk4 7000: GEAR-SONIC 학습 코드에 K1 overlay를 더해 학습했다. 계보는 `k1s_la4`(처음부터) → `k1s_walk` → `walk2` → `walk3` →
  `walk4` 7000 iter이고, NVIDIA가 공개한 GEAR-SONIC 가중치에서 시작하지 않았다. 학습은 [training.md](training.md), 평가는
  [results.md](results.md)에 있다.

## walk4 학습 데이터

[training.md §3](training.md#3-데이터)의 학습 세트(약 2.6시간):

- GEAR-SONIC 키네마틱 플래너 출력으로 만든 보행 클립. 플래너는 NVIDIA Open Model License이고, 이 라이선스는 출력을 파생 모델에서
  제외하며 출력에 소유권을 주장하지 않는다(§1.1, §2.4).
- GR00T-WholeBodyControl 예제 G1 모션을 K1으로 옮긴 클립.
- ROBOTIS cyclo_lab K1 댄스(Apache-2.0).
- 직접 만든 제스처·합성 서기 동작.
- 제3자 군무 영상("Guap")을 GEM-X로 리타깃한 12초 클립 하나와 그 미러(24초). 원본 영상의 권리는 원저작자에게 있고, 이 저장소와
  Release는 그 영상이나 모션을 배포하지 않는다.

BONES-SEED는 쓰지 않았다(BONES는 `k1s_bones*` 실험에만 썼다). SMPL-X 모델 파일이나 거기서 유도한 파일은 Release에 들어 있지 않다.

## 라이선스와 주의

- 가중치는 저장소 코드와 같은 [Apache License 2.0](../LICENSE)으로 배포한다. 제3자 고지는 [NOTICE](../NOTICE)와
  [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)에 있다.
- 이 저장소에서는 MuJoCo 시뮬레이션에서만 검증했다. 실제 로봇에 쓰는 것은 사용자 책임이다.
