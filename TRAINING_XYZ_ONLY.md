# TGCN-cVAE+MoE 학습 코드와 실행 기록 (XYZ-only)

이 문서는 기존 저장소의 `ray_in_dim=4` 설명과 구분하여, `END_IIW_ACCESS`에서 실행한 **ray XYZ 3차원** 학습을 추적할 수 있게 정리한 것입니다. 아래 코드는 2026-09-25에 WSL 원본 작업장에서 가져온 스냅샷입니다. 원본 작업장은 Git 저장소가 아니므로 과거 실행 시점의 소스 커밋을 증명할 수는 없습니다. 실행 로그와 체크포인트 해시를 대조하고, 현재 소스에서 두 체크포인트의 엄격한 로드와 출력 shape를 확인했습니다.

## 먼저 볼 파일

| 목적 | 파일 |
| --- | --- |
| TGCN 인코더와 MoE/접촉 헤드 | `proposed/TGCN_cVAE_MoE/model.py` |
| 프레임별 고정·동적 그래프 | `utils/graph.py` |
| 원본 열 선택과 파일 경계 시퀀스 | `utils/dataset.py` |
| 기본형 MSE+KL 학습 | `train_access_model.py` |
| 접촉 인지 학습 | `proposed/TGCN_cVAE_MoE/train.py` |
| 기본형이 포함된 실제 비교 실행기 | `run_train_all_comparison_models_xyz_only.sh` |
| 기본형과 접촉 인지 목적식 비교 실행기 | `run_train_tiara_objective_ablation_xyz_only.sh` |

`run_train_all_comparison_models_xyz_only.sh`는 여러 baseline을 순서대로 실행한 당시 명령의 기록입니다. 이 브랜치는 다른 baseline 소스를 갱신하지 않았으므로 TGCN만 재실행하려면 아래 기본형 명령을 사용하세요. 목적식 비교 실행기는 기존 기본형 체크포인트가 있으면 그것을 재사용하고 접촉 인지 모델을 학습합니다.

## 데이터 계약

각 프레임은 길이 10,464인 배열입니다. `utils/dataset.py`는 관절 특징 `0:135`를 `(15, 9)`, ray XYZ `204:1824`를 `(540, 3)`, 지도 IIW `7224:10464`를 `(6, 540)`으로 읽습니다. 따라서 배치 입출력은 각각 `(B,T,15,9)`, `(B,T,540,3)`, `(B,T,6,540)`입니다. ray의 네 번째 거리 특징이나 GT 레이블 열을 입력에 덧붙이지 않습니다.

학습·검증에는 `data/dataset_file_train.npy`, `data/dataset_file_val.npy`, `data/file_split_manifest.json`을 사용했습니다. 매니페스트의 `splits.train.file_ranges`와 `splits.val.file_ranges`는 시퀀스가 원본 파일 경계를 넘지 않게 합니다. 기록된 크기는 train 76,218프레임/358파일, validation 8,820프레임/43파일입니다. 학습 시퀀스 길이는 30, 시작 간격은 1이며 검증 시작 간격은 30입니다. 데이터와 체크포인트는 이 브랜치에 넣지 않았습니다. 원본 매니페스트 SHA-256: `e2a6560bc786c4d3294f962b8636b643edc911849352c48d19f4b6f35ed63b5b`.

## 기본형 TGCN: MSE + KL

`train_access_model.py`가 `IIWTGCN_cVAE`를 불러옵니다. 손실은 전체 IIW에 대한 평균제곱오차와 KL 항의 합입니다. KL은 배치 크기와 시간 길이의 곱으로 나눕니다. `beta = 0.01 × min(1, (epoch mod 10)/4)`인 10 epoch 주기를 사용합니다. AdamW 학습률 `1e-3`, weight decay `1e-4`, batch 16, 50 epochs, seed 3407, gradient norm 상한 5입니다. `best.pth`는 validation MSE가 최소인 시점에 저장합니다. AMP는 이 실행기에 없습니다.

원본 실행은 `run_train_all_comparison_models_xyz_only.sh`의 마지막 TGCN 단계였습니다. 같은 TGCN 단계만 실행하는 명령은 다음과 같습니다. 저장소 루트에서 실행하고, 위 세 데이터 파일을 먼저 배치해야 합니다.

```bash
python3 -u train_access_model.py \
  --model_name TGCN_cVAE_MoE_vanilla \
  --model_dir ./proposed/TGCN_cVAE_MoE \
  --class_name IIWTGCN_cVAE \
  --train_mmap_path ./data/dataset_file_train.npy \
  --val_mmap_path ./data/dataset_file_val.npy \
  --manifest_path ./data/file_split_manifest.json \
  --save_dir ./experiments/fair_architecture_xyz_only/TGCN_cVAE_MoE_vanilla_run1 \
  --epochs 50 --batch_size 16 --num_workers 8 --seed 3407
```

원본 `best.pth` SHA-256: `900b48236e6ec7b77674ba62a382f4cd13a7f2dc9a539269951544488ebb2c8f`.

## 접촉 인지 TIARA TGCN: 세 지도 손실 + KL

`proposed/TGCN_cVAE_MoE/train.py`의 목적식은 접촉 가중 MSE, 활성 위치 Smooth L1, 접촉 Focal BCE, free-bits KL로 구성됩니다. 기본 가중치는 활성 손실 `2.0`, 접촉 손실 `0.5`, KL의 최대 `beta=0.001`입니다. KL beta는 처음 10 epochs 동안 단조 증가합니다. GT 접촉은 `IIW > 0`, 검증 접촉 예측은 `IIW >= 0.1`로 계산합니다. 부위별 가중치는 train 데이터의 활성 ray 빈도에서 자동 산출되고, 접촉 균형 시퀀스 샘플링을 사용합니다. AdamW, batch 16, 50 epochs, seed 3407, CUDA AMP를 사용했습니다. `best.pth`는 실행기가 전달한 `--checkpoint_metric recon`에 따라 **접촉 가중 validation MSE**가 최소인 시점에 저장했습니다. 검증 F1 기준으로 고른 체크포인트가 아닙니다.

코드는 `0.02 × gate_balance_loss`도 총손실의 스칼라 값에 더합니다. 그러나 `model.py`에서 저장한 `last_gate_weights`가 `detach()`되어 있으므로 이 항은 gate에 gradient를 전달하지 않습니다. 따라서 이 실행 결과를 gate 균형 정규화의 학습 효과로 해석하면 안 됩니다. 실제로 최적화되는 항은 위 세 지도 손실과 KL입니다.

원본 목적식 비교 실행은 `run_train_tiara_objective_ablation_xyz_only.sh`로 수행했습니다. 접촉 인지 단계만 실행하는 동등한 명령은 다음과 같습니다.

```bash
python3 -u proposed/TGCN_cVAE_MoE/train.py \
  --epochs 50 --batch_size 16 --num_workers 8 --seed 3407 --amp \
  --train_mmap_path ./data/dataset_file_train.npy \
  --val_mmap_path ./data/dataset_file_val.npy \
  --manifest_path ./data/file_split_manifest.json \
  --train_manifest_split train --val_manifest_split val \
  --checkpoint_metric recon \
  --save_dir ./experiments/objective_ablation_xyz_only/TIARA_TGCN_cVAE_MoE_contact_aware_run1
```

원본 `best.pth` SHA-256: `c0452e23a8f11799bb5d01cde4368f3479a821a3a254c7d4c2371a17a7d61efb`.

## 코드 스냅샷 SHA-256

| 파일 | SHA-256 |
| --- | --- |
| `proposed/TGCN_cVAE_MoE/model.py` | `69d40842ea46706bea5aa18277f5599706c435da683516556da3a6d96afdf1db` |
| `proposed/TGCN_cVAE_MoE/train.py` | `d5be435729e3293d2c853783c662c88ce04a0f508862449466a19c9e8b6d3cc4` |
| `utils/dataset.py` | `372e36fb92815d3d5fe3e66947fa3b9902654b9c817d385a8e9ba9935ce170b5` |
| `utils/graph.py` | `d5b9fa3315019d4d3580a7dcb71b06f1eced5944dfe76bc7ca5894f3867de6cf` |
| `train_access_model.py` | `7faeda0ba557f8069783734f2af92e7eb1bb0b556024a65c89a7c5c5aa998f20` |

## 확인한 범위

- 다섯 Python 파일의 구문 컴파일, 두 Bash 실행기의 구문 검사, 두 학습 CLI의 `--help`가 통과했습니다.
- 원본 기본형 및 접촉 인지 `best.pth`를 현재 `IIWTGCN_cVAE`에 각각 `strict=True`로 로드하고, 작은 CPU 입력에서 `(1,2,6,540)` 예측 shape를 확인했습니다.
- 파일 경계가 있는 합성 프레임에서 30프레임 창과 `(15,9)/(540,3)/(6,540)` 특징 shape를 확인했습니다.
- gate balance 스칼라는 `requires_grad=False`로 확인했습니다.

이 코드는 학습 방법을 검토하고, 원본 데이터가 있을 때 다시 실행할 수 있게 제공합니다. 같은 가중치나 동일한 지표가 재현된다는 주장은 실행 환경, 원본 데이터, 매니페스트, 체크포인트와 전체 평가 절차를 대조한 뒤에만 할 수 있습니다.
