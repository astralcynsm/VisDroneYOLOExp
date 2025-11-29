import torch
from pathlib import Path
from ultralytics import YOLO
import csv
import argparse

# --- 0. 命令行参数设置 ---
parser = argparse.ArgumentParser(description="Standard YOLOv8 prediction with structured CSV output.")
parser.add_argument('--filename', type=str, required=True, help='Filename of the image to process.')
args = parser.parse_args()

# --- 1. 配置与路径管理 ---
BASE_IMAGE_PATH = Path('/mnt/Storage/files/MachineLearning/VisDroneDataset/Datasets/VisDrone2019-DET-train/images/')
YOLO_MODEL_PATH = '/mnt/Storage/files/MachineLearning/VisDroneDataset/runs/from_colab/detect/visdrone_yolov8l_A100_colab/weights/best.pt'
TEST_IMAGE_PATH = BASE_IMAGE_PATH / args.filename

if not TEST_IMAGE_PATH.is_file():
    print(f"Error: Image file not found at '{TEST_IMAGE_PATH}'")
    exit()

OUTPUT_DIR = Path('yolo_standard_results/')
OUTPUT_DIR.mkdir(exist_ok=True)

# --- 2. 加载模型并执行标准推理 ---
print(f"Loading model and running standard YOLOv8 prediction on '{args.filename}'...")
model = YOLO(YOLO_MODEL_PATH)

# 执行推理，同时保存带标注的图片
results = model.predict(
    source=str(TEST_IMAGE_PATH),
    conf=0.35, # 和 SAHI 脚本保持一致的置信度阈值
    save=True, # 保存带标注的图片
    project=str(OUTPUT_DIR),
    name=f"annotated_{Path(TEST_IMAGE_PATH).stem}"
)

print("Prediction completed. Processing results...")

# --- 3. 提取数据并保存为 CSV ---
# `results` 是一个列表，我们处理第一张图的结果
result = results[0]
boxes = result.boxes  # Boxes 对象包含所有检测框信息
predictions_data = []

for i in range(len(boxes)):
    box_data = boxes[i]
    class_id = int(box_data.cls)
    predictions_data.append({
        'id': i + 1,
        'class_name': model.names[class_id],
        'confidence': float(box_data.conf),
        'bbox': box_data.xyxy[0].cpu().numpy().tolist() # .tolist() 方便写入CSV
    })

csv_path = OUTPUT_DIR / f"standard_result_data_{Path(TEST_IMAGE_PATH).stem}.csv"
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'Class', 'Confidence', 'Bbox_XYXY'])
    for data in predictions_data:
        writer.writerow([
            data['id'],
            data['class_name'],
            f"{data['confidence']:.4f}",
            [round(coord, 2) for coord in data['bbox']] # 坐标保留两位小数
        ])

print(f"\nSuccess! Found {len(predictions_data)} objects.")
print(f"  - Annotated image saved in: {OUTPUT_DIR / f'annotated_{Path(TEST_IMAGE_PATH).stem}'}")
print(f"  - Structured data saved to: {csv_path}")
