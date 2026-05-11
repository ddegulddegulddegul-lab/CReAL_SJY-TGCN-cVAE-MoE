# TGCN-cVAE+MoE 기반 3D 인체-가구 상호작용(IIW) 예측 모델

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)
![Viser](https://img.shields.io/badge/Viser-3D_Viz-orange.svg)

본 저장소는 **적응형 시공간 그래프 합성곱 기반의 전문가 혼합(MoE) 구조 조건부 VAE (TGCN-cVAE+MoE)** 의 공식 파이토치(PyTorch) 구현체입니다. 
이 딥러닝 모델은 시뮬레이터 상에서의 광선 추적(Ray-Casting) 기법을 활용하여, 3D 인체 관절(Joint)과 주변 가구 환경 사이호의 **상호작용 강도 가중치(Interaction Intensity Weight: IIW)** 를 프레임 단위로 예측합니다.

## 🚀 프로젝트 주요 특징 (Features)
- **동적 이분 그래프 (Dynamic Bipartite Graph)**: 프레임마다 변하는 인체의 15개 관절과 주변 환경으로 뻗어나간 540개의 광선 노드 사이의 거리를 **역 유클리드 거리 (Inverse Euclidean Distance)** 로 계산하여 동적인 그래프 인접 행렬을 구축합니다.
- **TGCN 인코더 (Encoder)**: 공간적 상관관계(Adaptive GCN)와 시간적 연속성(Dilated TCN)을 동시에 포착하여 부드럽고 물리적으로 안정적인 상호작용 피처를 추출합니다.
- **MoE 디코더 (Mixture of Experts Decoder)**: 상호작용(접촉)이 발생하는 데이터가 전체의 5% 미만인 데이터 불균형(Sparsity) 문제를 극복하기 위해, 3개의 개별 전문가 신경망을 분업화하여 IIW를 정밀하게 생성합니다.
- **주기적 KL 어닐링 (Cyclical KL Annealing)**: 희소한 데이터셋에서 자주 발생하는 **사후 분포 붕괴 (Posterior Collapse)** 문제를 방지하고 표현 다양성을 보장합니다.

## 📁 폴더 및 파일 구조 (Directory Structure)

```text
.
├── baselines/                 # 비교 및 절제 실험용 오픈소스 모델
│   ├── MLP_cVAE_MoE/          # 가장 단순한 MLP 형태
│   ├── LSTM_cVAE_MoE/         # 시간축 모델링 비교용 (Sequence RNN)
│   ├── GCN_cVAE_MoE/          # 공간축 모델링 비교용 (Spatial Graph)
│   ├── TCN_cVAE_MoE/          # 시간축 모델링 비교용 (Temporal Conv)
│   └── TGCN_cVAE_TGCN/        # 제안 모델 디코더 평가용 (MoE 대신 TGCN 디코더)
│
├── proposed/                  # ⭐ 제안 모델 폴더 (Ours)
│   └── TGCN_cVAE_MoE/
│       ├── model.py           # 논문에 언급된 핵심 아키텍처
│       ├── train.py           # TensorBoard 로깅이 포함된 학습 스크립트
│       └── eval.py            # 모델 평가 스크립트 (MAE, Active MAE, Jittering, FLOPs)
│
├── utils/                     # 공용 서브 모듈
│   ├── dataset.py             # Memory-mapped DataLoader 생성 및 데이터 분할(8:1:1 split)
│   └── graph.py               # 동적 이분 그래프 인접 행렬 구축 클래스
│
├── data/                      # 최종 무결성 평가용으로 격리된 데이터 폴더
├── Extraction/                # 전처리(mmap) 대상이 되는 원본 데이터 모음
├── visualize.py               # 3D 인터랙티브 GUI 시각화 스크립트 (Viser)
└── run_all_baselines.sh       # 모든 베이스라인 모델 순차 자동 학습 쉘 스크립트
```

## ⚙️ 설치 및 세팅 (Installation)

1. 깃허브 저장소를 클론합니다:
```bash
git clone https://github.com/사용자아이디/레포지토리이름.git
cd 레포지토리이름
```

2. 필수 라이브러리를 설치합니다:
```bash
pip install -r requirements.txt
```

## 🏋️‍♂️ 모델 학습 (Training)

학습을 시작하기 전, 용량 문제로 깃허브에 업로드되지 않은 `dataset_mmap.npy` 파일이 루트 폴더에 준비되어 있어야 합니다.

제안 모델 학습 커맨드:
```bash
python proposed/TGCN_cVAE_MoE/train.py \
    --mmap_path ./dataset_mmap.npy \
    --save_dir ./proposed/TGCN_cVAE_MoE/weights \
    --epochs 50 \
    --batch_size 32 \
    --num_workers 8
```
학습 상태는 텐서보드(TensorBoard)를 통해 실시간으로 모니터링할 수 있습니다.
```bash
tensorboard --logdir ./proposed/TGCN_cVAE_MoE/weights/logs
```

## 📊 모델 평가 (Evaluation)

학습에 사용하지 않은 held-out 테스트셋(Test-split) 위에서 성능을 평가합니다. Jittering 등을 포함하여 논문에 표기된 핵심 성능 지표들을 출력합니다.
```bash
python proposed/TGCN_cVAE_MoE/eval.py \
    --mmap_path ./dataset_mmap.npy \
    --weights_path ./proposed/TGCN_cVAE_MoE/weights/best.pth \
    --batch_size 16
```

## 🎮 3D 시각화 도구 (Visualization)

학습이 끝난 모델이 인체의 어떤 부위와 어떤 가구의 접촉(IIW)을 예측하는지 3D 공간 상에서 직관적으로 확인할 수 있는 `viser` 기반 웹 시각화 툴을 제공합니다.

```bash
python proposed/TGCN_cVAE_MoE/visualize.py \
    --weights_path ./proposed/TGCN_cVAE_MoE/weights/best.pth \
    --data_dir ./data/ \
    --port 8080
```
터미널 실행 후 브라우저에서 **`http://localhost:8080`** 에 접속하세요.
- 화면 내 컨트롤러를 이용해 시퀀스 재생/정지 및 특정 타임 프레임을 탐색할 수 있습니다.
- 특정 신체 부위(예: 골반, 척추, 양 손, 양 발)만 개별적으로 선택하여 해당 부위에 매핑되는 540개 광선(Ray)의 예측 강도를 히트맵 스타일로 확인할 수 있습니다.
