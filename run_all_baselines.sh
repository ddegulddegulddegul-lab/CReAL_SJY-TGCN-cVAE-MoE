#!/bin/bash

# 자동화된 리눅스 백그라운드 학습 스크립트 (50 Epoch)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$PROJECT_ROOT/baselines_training.log"
BASELINES=("MLP_cVAE_MoE" "LSTM_cVAE_MoE" "TCN_cVAE_MoE" "GCN_cVAE_MoE" "TGCN_cVAE_TGCN")

cd "$PROJECT_ROOT" || exit 1

echo "==========================================================" > "$LOG_FILE"
echo "Starting Bulk Training & Evaluation at $(date)" >> "$LOG_FILE"
echo "==========================================================" >> "$LOG_FILE"

for BASE in "${BASELINES[@]}"; do
    echo "" >> "$LOG_FILE"
    echo "==========================================================" >> "$LOG_FILE"
    echo "🚀 [$(date)] Starting Training for $BASE " >> "$LOG_FILE"
    echo "==========================================================" >> "$LOG_FILE"
    
    TARGET_DIR="$PROJECT_ROOT/baselines/$BASE"
    WEIGHTS_DIR="$TARGET_DIR/weights"
    
    # 가중치 폴더 생성 및 기존 파일 강제 초기화
    mkdir -p "$WEIGHTS_DIR"
    rm -rf "$WEIGHTS_DIR"/*
    
    # 50 Epoch 본학습 (num_workers=8 최고 속도)
    python "$TARGET_DIR/train.py" --epochs 50 --batch_size 16 --num_workers 8 --save_dir "$WEIGHTS_DIR/" >> "$LOG_FILE" 2>&1
    
    echo "----------------------------------------------------------" >> "$LOG_FILE"
    echo "📊 [$(date)] Evaluating $BASE " >> "$LOG_FILE"
    echo "----------------------------------------------------------" >> "$LOG_FILE"
    
    # 학습 종료 즉시 평가 지표(RMSE, Active MAE, Jitter 등) 추출
    python "$TARGET_DIR/eval.py" --batch_size 16 >> "$LOG_FILE" 2>&1
    
done

echo "" >> "$LOG_FILE"
echo "==========================================================" >> "$LOG_FILE"
echo "✅ All baseline trainings and evaluations successfully completed at $(date)!" >> "$LOG_FILE"
echo "==========================================================" >> "$LOG_FILE"
