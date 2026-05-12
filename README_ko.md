# IIW Ver2: 접촉 인식 TGCN-cVAE-MoE

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)
![Viser](https://img.shields.io/badge/Viser-3D_Viz-orange.svg)

이 브랜치는 `iiw_ver2` 실험 버전입니다. 목표는 사람의 모션 시퀀스와
가구 ray 샘플 사이의 Interaction Intensity Weight, 즉 IIW를 예측하는
것입니다.

ver2는 초기 버전에서 보였던 문제를 줄이는 데 초점을 둡니다. 전체 MSE가
쉬운 접촉이나 비접촉 ray에 의해 지배되면, 손이나 허리처럼 희소하거나
어려운 접촉이 과소 예측될 수 있습니다. ver2는 이 문제를 contact 중심
loss와 sampling으로 보완합니다.

중요하게, 이 모델은 가구 종류 라벨을 입력으로 사용하지 않습니다. 원본
feature vector 안에 있는 가구 class label 영역은 모델 입력에서 제외됩니다.

## 현재 포함된 학습 모델

이 브랜치에는 contact-F1 run에서 얻은 best checkpoint가 포함되어 있습니다.

- 모델: `proposed/TGCN_cVAE_MoE`
- 체크포인트: `proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/best.pth`
- 선택 기준: validation contact micro F1
- best epoch: 38
- best validation contact micro F1: 0.955008
- best validation contact macro F1: 0.904828
- 학습 설정: 50 epochs

validation 성능은 epoch 38에서 가장 좋았고, 이후 epoch에서는 contact-F1이
더 개선되지 않았습니다. 따라서 평가와 시각화에는 `best.pth`를 사용하는
것을 기준으로 합니다.

## Ver1에서 달라진 점

- random frame split 대신 scene-level split을 사용합니다.
- body-part별 접촉 분포 차이를 줄이기 위해 contact-stratified scene split을 제공합니다.
- 접촉이 포함된 temporal window를 더 자주 보도록 contact-balanced sampling을 사용합니다.
- decoder가 contact logits와 IIW intensity를 분리해서 예측합니다.
- 최종 IIW는 `sigmoid(contact_logits) * sigmoid(intensity_logits)`로 계산합니다.
- loss는 weighted reconstruction, active-region SmoothL1, focal contact loss,
  KL free-bits, KL warmup, MoE gate-balance term을 함께 사용합니다.
- best checkpoint는 MSE가 아니라 validation contact F1 기준으로 선택합니다.
- 평가 지표에 contact precision, recall, F1, body-part별 지표를 추가했습니다.

## 예측 부위

모델은 6개 신체 부위에 대해 IIW를 예측합니다.

1. pelvis/base
2. spine
3. right hand
4. left hand
5. right foot
6. left foot

각 frame에서 출력은 540개 가구/environment ray에 대한 `6 x 540` IIW field입니다.

## 폴더 구조

```text
.
├── proposed/
│   └── TGCN_cVAE_MoE/
│       ├── model.py
│       ├── train.py
│       ├── eval.py
│       ├── visualize.py
│       └── weights_ver2_contact_f1_run2/
│           ├── best.pth
│           ├── training_metrics.csv
│           └── training_metrics.png
├── baselines/
│   ├── MLP_cVAE_MoE/
│   ├── LSTM_cVAE_MoE/
│   ├── GCN_cVAE_MoE/
│   ├── TCN_cVAE_MoE/
│   └── TGCN_cVAE_TGCN/
├── scene_splits/
├── scene_splits_contact_stratified/
├── utils/
├── experiments/
├── scripts/
├── TRAINED_MODEL_VER2.md
└── VERSION2_NOTES.md
```

다음 대용량 로컬 데이터는 GitHub에 포함하지 않습니다.

- `Extraction/`
- `data/`
- `dataset_mmap.npy`
- `scene_splits*/dataset_scene_*.npy`
- 중간 checkpoint와 log 파일

대신 split manifest와 scene list를 포함해 같은 split을 다시 만들 수 있게 했습니다.

## 데이터 준비

로컬에 raw extraction 파일이 있다면 contact-stratified scene split은 다음 명령으로
다시 만들 수 있습니다.

```bash
python3 utils/build_contact_stratified_scene_split_mmaps.py --overwrite
```

사용한 split 요약은 다음과 같습니다.

- Train: 7 scenes, 67,738 frames, 76.81%
- Validation: 2 scenes, 10,542 frames, 11.95%
- Test: 2 scenes, 9,905 frames, 11.23%

최종 정성 검증용 `data/` 폴더는 모델 선택 이후 별도로 사용하는 외부 데이터로
분리합니다.

## 설치

```bash
git clone https://github.com/ddegulddegulddegul-lab/CReAL_SJY-TGCN-cVAE-MoE.git
cd CReAL_SJY-TGCN-cVAE-MoE
git checkout iiw_ver2
pip install -r requirements.txt
```

## 학습

ver2의 기본 학습 스크립트는 다음과 같습니다.

```bash
bash scripts/run_train_contact_f1.sh
```

동일한 명령을 풀어서 쓰면 다음과 같습니다.

```bash
python3 -u proposed/TGCN_cVAE_MoE/train.py \
  --epochs 50 \
  --batch_size 16 \
  --num_workers 8 \
  --amp \
  --save_dir ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2
```

학습 기본값은 다음 split을 사용합니다.

- `./scene_splits_contact_stratified/dataset_scene_train.npy`
- `./scene_splits_contact_stratified/dataset_scene_val.npy`
- checkpoint metric: `contact_f1`

epoch별 학습 로그는 save directory의 `training_metrics.csv`에 저장됩니다.

학습 그래프를 다시 만들려면 다음 명령을 사용합니다.

```bash
python3 utils/plot_training_metrics.py \
  --csv_path ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/training_metrics.csv \
  --output_path ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/training_metrics.png
```

## 평가

포함된 best checkpoint를 contact-stratified scene test set에서 평가합니다.

```bash
python3 proposed/TGCN_cVAE_MoE/eval.py \
  --weights_path ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/best.pth \
  --batch_size 16
```

빠르게 non-overlapping 방식으로 확인하려면 다음처럼 실행할 수 있습니다.

```bash
python3 proposed/TGCN_cVAE_MoE/eval.py \
  --weights_path ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/best.pth \
  --batch_size 16 \
  --stride 60
```

## 시각화

`viser` 기반 viewer로 `data/` 폴더의 정성 검증 시퀀스에 대해 IIW 예측을 확인할 수 있습니다.

```bash
python3 proposed/TGCN_cVAE_MoE/visualize.py \
  --data_dir ./data \
  --weights_path ./proposed/TGCN_cVAE_MoE/weights_ver2_contact_f1_run2/best.pth \
  --port 8081
```

브라우저에서 `http://localhost:8081`로 접속합니다.

`All Parts` 모드의 색상 규칙은 다음과 같습니다.

- pelvis/base: 빨강
- spine: 보라
- hands: 초록
- feet: 파랑

현재 viewer에는 예전 spine 강조 override가 없습니다. `All Parts` 모드에서는
각 ray마다 가장 높은 IIW를 예측한 신체 부위의 색상만 표시합니다.
