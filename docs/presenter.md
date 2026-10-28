# Linux 한 대에서 SHAPE Dual Stage 실행

카메라 → MediaPipe Pose full → 팔 방향 IK → PD 팔 제어 → MuJoCo K1을 모두 같은 PC에서 실행합니다.
현재 발표용 설정은 **팔 모방 + 서 있기**입니다. 다리 자세를 사람과 똑같이 따라 하는 전신 모방은 포함하지 않습니다.
관객 화면은 검은 배경에 HUMAN / K1 두 영상을 배치하고 SHAPE와 학교 이름을 표시합니다.
SNU 정문은 렌더링 전용 배경이며 로봇의 충돌·질량·제어 모델을 바꾸지 않습니다.

## 설치

Ubuntu x86_64, Python 3.12, NVIDIA 드라이버, 연결된 USB 웹캠을 기준으로 합니다.
MediaPipe와 ONNX 제어는 CPU, MuJoCo 렌더링은 EGL을 사용합니다. CUDA Toolkit이나 TensorRT 설치는 필요하지 않습니다.
기존 가상환경이 있다면 별도 clone에서 설치하는 편이 기존 실행 환경을 보존하기 좋습니다.

```bash
sudo apt install git curl python3.12-venv libgles2 libegl1 libgl1
git clone https://github.com/HIWooI/k1-sim-teleop.git
cd k1-sim-teleop
PYTHON=python3.12 bash scripts/setup_presenter.sh
bash run_presenter.sh --source 0
```

설치 스크립트는 전용 의존성, 공개 ROBOTIS K1 자산, 공개 K1 정책, MediaPipe full 모델을 받습니다.
정책과 MediaPipe 모델의 SHA256을 확인합니다. Torch, G1 자산, GEM-X, 비공개 전신 정책은 필요하지 않습니다.
이미 받은 ROBOTIS 자산은 설치와 실행 양쪽에 `AI_SAPIENS_ROOT=/절대/경로`를 지정해서 재사용할 수 있습니다.

- 관객 화면: <http://127.0.0.1:8700/display> — 실행 시 브라우저 자동 열기. `F` 또는 더블클릭으로 전체 화면.
- 운영 패널: <http://127.0.0.1:8700/> — `C` 보정, `Space` 시작/중지, `R` 리셋, `Esc` 비상정지.
- 종료: 실행한 터미널에서 `Ctrl+C`. 기본 실행 시간 제한은 없습니다.
- 진단 로그: `logs/k1live/<날짜-시간>/`. 카메라 또는 모델 오류는 패널과 로그에 표시됩니다.

이 실행기는 매번 `k1live/web/presenter.yaml`의 검증된 설정으로 시작합니다. 예전 패널에 저장된 hybrid/GEM-X 등의 선택을
불러오지 않습니다. 실행 중 패널에서 바꾼 선택은 `presenter-settings.json`에만 기록되며 일반 서비스 설정을 덮어쓰지 않습니다.

## 발표 전 리허설

카메라에서 한 사람의 머리와 양팔이 모두 보이게 서고, 정면을 본 채 팔을 편하게 내려 보정합니다.
자동 보정/추종이 켜져 있지만 중립 자세가 어긋나면 운영 패널에서 `C`로 다시 보정합니다.
인물이 여러 명이면 패널의 사람 찾을 영역을 지정하거나 `--crop 0.2,0,0.8,1`로 발표자 영역을 제한합니다.

```bash
# 실제 카메라 대신 영상 파일을 카메라 속도로 반복 재생 (실시간 인식과 물리 제어는 동일)
bash run_presenter.sh --source /absolute/path/presenter.mp4

# 정문 없는 기존 배경 / 브라우저 자동 열기 끄기
bash run_presenter.sh --scene plain --no-browser

# 다른 서비스와 동시에 실행: 관객 화면 http://127.0.0.1:8800/display
bash run_presenter.sh --port-offset 100
```

카메라가 열리지 않으면 다른 앱의 카메라 사용을 종료하고 운영 패널에서 카메라 번호를 바꿉니다.
Linux의 `/dev/video*` 접근 권한도 확인합니다. EGL 오류는 먼저 NVIDIA 드라이버와 `libegl1`, `libgles2`를 확인합니다.
원격 SSH에서 브라우저가 없는 경우 `--no-browser`를 사용합니다.

팔 필터 6 Hz / beta 2, 속도·가속도 제한 18 rad/s / 300 rad/s²를 사용합니다.
화면은 최신 JPEG 두 장을 받은 뒤 함께 교체하며 오래된 화면을 쌓지 않습니다.
두 카메라 촬영 시각의 정확한 동기화나 종단 지연 수치를 보장하지는 않습니다.

## 검증 범위

Linux PRO 6000에서 영상 입력 → 실제 MediaPipe 인식 → split 제어 → EGL 렌더링 경로를 확인합니다.
RTX 5080 PC의 물리 웹캠 연결과 현장 조명·거리·USB 장치 조건은 해당 PC에서 리허설이 필요합니다.
원격 PRO 6000 접속이나 Mac의 경로는 이 실행기에 필요하지 않습니다.

배경 자산과 폰트 출처는 [presenter-assets.md](presenter-assets.md)를 참고합니다.
