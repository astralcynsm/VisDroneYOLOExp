import cv2
import torch
import numpy as np
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
import argparse
import csv

# --- 0. 命令行参数 ---
parser = argparse.ArgumentParser(description="Ultimate tool to compare SAHI and Standard YOLOv8 predictions.")
parser.add_argument('--filename', type=str, required=True, help='Filename of the image to process.')
args = parser.parse_args()

# --- 1. 配置与路径 ---
BASE_IMAGE_PATH = Path('/mnt/Storage/files/MachineLearning/VisDroneDataset/Datasets/VisDrone2019-DET-train/images/')
YOLO_MODEL_PATH = '/mnt/Storage/files/MachineLearning/VisDroneDataset/runs/from_colab/detect/visdrone_yolov8l_A100_colab/weights/best.pt'
TEST_IMAGE_PATH = BASE_IMAGE_PATH / args.filename
IOU_THRESHOLD = 0.5 # IoU 匹配阈值

if not TEST_IMAGE_PATH.is_file(): exit(f"Error: Image file not found at '{TEST_IMAGE_PATH}'")

# --- 2. 辅助函数 ---
def calculate_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def run_sahi_prediction(model_path, image_path):
    detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=model_path, confidence_threshold=0.35, device="cuda")
    sahi_result = get_sliced_prediction(image=str(image_path), detection_model=detection_model, slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
    return [{'class_name': p.category.name, 'confidence': p.score.value, 'bbox': p.bbox.to_xyxy()} for p in sahi_result.object_prediction_list]

def run_standard_prediction(model, image_path):
    results = model.predict(source=str(image_path), conf=0.35)
    result = results[0]
    boxes = result.boxes
    return [{'class_name': model.names[int(b.cls)], 'confidence': float(b.conf), 'bbox': b.xyxy[0].cpu().numpy().tolist()} for b in boxes]

def draw_results(image, predictions, title):
    for pred in predictions:
        box = [int(c) for c in pred['bbox']]
        label = f"{pred['class_name']}: {pred['confidence']:.2f}"
        color = pred['color']
        # 绘制更细的框
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness=2)
        # 绘制更小的、带背景的文字
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (box[0], box[1] - h - 10), (box[0] + w, box[1] - 5), color, -1)
        cv2.putText(image, label, (box[0], box[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image

# --- 3. 主流程 ---
# 加载模型
yolo_model = YOLO(YOLO_MODEL_PATH)

# 分别执行两种预测
print("Running SAHI prediction...")
sahi_preds = run_sahi_prediction(YOLO_MODEL_PATH, TEST_IMAGE_PATH)
print(f"  -> Found {len(sahi_preds)} objects.")

print("Running Standard YOLO prediction...")
standard_preds = run_standard_prediction(yolo_model, TEST_IMAGE_PATH)
print(f"  -> Found {len(standard_preds)} objects.")

# IoU 匹配
print("Matching predictions using IoU...")
matches = []
sahi_indices_matched = set()
standard_indices_matched = set()

for i, sahi_pred in enumerate(sahi_preds):
    best_iou = 0
    best_j = -1
    for j, std_pred in enumerate(standard_preds):
        if j in standard_indices_matched: continue
        iou = calculate_iou(sahi_pred['bbox'], std_pred['bbox'])
        if iou > best_iou:
            best_iou = iou
            best_j = j
    
    if best_iou > IOU_THRESHOLD:
        matches.append({'sahi': sahi_pred, 'standard': standard_preds[best_j], 'iou': best_iou})
        sahi_indices_matched.add(i)
        standard_indices_matched.add(best_j)

sahi_only = [sahi_preds[i] for i in range(len(sahi_preds)) if i not in sahi_indices_matched]
standard_only = [standard_preds[i] for i in range(len(standard_preds)) if i not in standard_indices_matched]

print(f"  -> Matched pairs: {len(matches)}")
print(f"  -> SAHI only detections: {len(sahi_only)}")
print(f"  -> Standard YOLO only detections: {len(standard_only)}")


# --- 4. 生成输出 ---
# 准备输出目录
run_id = Path(args.filename).stem
output_dir = Path(f'comparison_results/{run_id}')
output_dir.mkdir(parents=True, exist_ok=True)

# 写入CSV报告
csv_path = output_dir / 'comparison_report.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Type', 'Class', 'SAHI_Confidence', 'Standard_Confidence', 'IoU', 'SAHI_Bbox', 'Standard_Bbox'])
    for match in matches:
        writer.writerow(['Matched', match['sahi']['class_name'], f"{match['sahi']['confidence']:.4f}", f"{match['standard']['confidence']:.4f}", f"{match['iou']:.4f}", match['sahi']['bbox'], match['standard']['bbox']])
    for pred in sahi_only:
        writer.writerow(['SAHI_Only', pred['class_name'], f"{pred['confidence']:.4f}", 'N/A', 'N/A', pred['bbox'], 'N/A'])
    for pred in standard_only:
        writer.writerow(['Standard_Only', pred['class_name'], 'N/A', f"{pred['confidence']:.4f}", 'N/A', 'N/A', pred['bbox']])
print(f"  -> Comparison report saved to: {csv_path}")


# 生成可视化对比图
# 颜色定义: 绿色=匹配, 蓝色=SAHI独有, 红色=YOLO独有
COLOR_MATCHED = (0, 255, 0) # Green
COLOR_SAHI_ONLY = (255, 0, 0) # Blue
COLOR_YOLO_ONLY = (0, 0, 255) # Red

# 准备 SAHI 视图的数据
sahi_view_preds = []
for match in matches: sahi_view_preds.append({**match['sahi'], 'color': COLOR_MATCHED})
for pred in sahi_only: sahi_view_preds.append({**pred, 'color': COLOR_SAHI_ONLY})
# 准备 Standard 视图的数据
standard_view_preds = []
for match in matches: standard_view_preds.append({**match['standard'], 'color': COLOR_MATCHED})
for pred in standard_only: standard_view_preds.append({**pred, 'color': COLOR_YOLO_ONLY})

# 绘制并保存
image_sahi = cv2.imread(str(TEST_IMAGE_PATH))
image_std = image_sahi.copy()

sahi_img_path = output_dir / 'comparison_SAHI_view.png'
std_img_path = output_dir / 'comparison_Standard_view.png'

draw_results(image_sahi, sahi_view_preds, "SAHI Predictions")
draw_results(image_std, standard_view_preds, "Standard Predictions")

cv2.imwrite(str(sahi_img_path), image_sahi)
cv2.imwrite(str(std_img_path), image_std)
print(f"  -> Visual comparison images saved to: {output_dir}")
