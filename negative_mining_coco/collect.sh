#!/bin/bash

# --- 配置 ---
# yolo 会自动创建 predict 目录，如果有多个，选择最新的 predict, predict2 ...
PREDICT_LABEL_DIR="runs/detect/predict/labels"
SOURCE_IMG_DIR="unlabeled2017"
OUTPUT_DIR="hard_negatives"
NUM_SAMPLES=2000

# --- 主体 ---
echo "开始收集困难负样本..."

# 清理旧目录（如果存在）
rm -rf "${OUTPUT_DIR}"

# 创建输出目录结构
mkdir -p "${OUTPUT_DIR}/images"
mkdir -p "${OUTPUT_DIR}/labels"

echo "从 ${PREDICT_LABEL_DIR} 目录中随机抽取 ${NUM_SAMPLES} 个样本..."

# 找到所有非空的预测结果，打乱顺序，然后取前 N 个
find "${PREDICT_LABEL_DIR}" -type f -size +0c | shuf -n ${NUM_SAMPLES} | while read -r label_file; do

  filename=$(basename "${label_file}" .txt)

  # 1. 拷贝原始图片到 'images' 目录
  cp "${SOURCE_IMG_DIR}/${filename}.jpg" "${OUTPUT_DIR}/images/"

  # 2. 在 'labels' 目录中创建一个对应的空文件
  touch "${OUTPUT_DIR}/labels/${filename}.txt"

done

echo "-----------------------------------"
echo "成功收集 ${NUM_SAMPLES} 个困难负样本到 ${OUTPUT_DIR} 目录！"
