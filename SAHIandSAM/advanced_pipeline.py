import cv2
import torch
import numpy as np
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import csv
import argparse
import shutil

# --- 0. 命令行参数设置 ---
parser = argparse.ArgumentParser(description="Professional pipeline for Detection and Segmentation with structured, functional output.")
parser.add_argument('--filename', type=str, required=True, help='Filename of the image to process (must be in the base image directory).')
args = parser.parse_args()

# --- 1. 配置与路径管理 ---
BASE_IMAGE_PATH = Path('/mnt/Storage/files/MachineLearning/VisDroneDataset/Datasets/VisDrone2019-DET-train/images/')
YOLO_MODEL_PATH = '/mnt/Storage/files/MachineLearning/VisDroneDataset/runs/from_colab/detect/visdrone_yolov8l_A100_colab/weights/best.pt'
SAM_CHECKPOINT_PATH = 'sam_vit_b_01ec64.pth'
SAM_MODEL_TYPE = "vit_b"
TEST_IMAGE_PATH = BASE_IMAGE_PATH / args.filename # 自动拼接完整路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 检查输入文件是否存在
if not TEST_IMAGE_PATH.is_file():
    print(f"Error: Image file not found at '{TEST_IMAGE_PATH}'")
    exit()

# 结构化输出目录管理
BASE_OUTPUT_DIR = Path('results/')
CONFIDENCE_DIR = BASE_OUTPUT_DIR / 'confidence'
SQUARE_ONLY_DIR = BASE_OUTPUT_DIR / 'squareOnly'
ORIGINAL_DIR = BASE_OUTPUT_DIR / 'original'
for d in [CONFIDENCE_DIR, SQUARE_ONLY_DIR, ORIGINAL_DIR]: d.mkdir(parents=True, exist_ok=True)

# 自动确定唯一ID
existing_ids = [int(p.stem) for p in ORIGINAL_DIR.glob('*.jpg')]
run_id = max(existing_ids) + 1 if existing_ids else 1
print(f"Assigning new Run ID: {run_id}")

# --- 辅助函数 ---
def show_mask(mask, ax, random_color=False):
    if random_color: color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else: color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]; mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1); ax.imshow(mask_image)
def show_box(box, ax):
    x0, y0 = box[0], box[1]; w, h = box[2] - x0, box[3] - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='lime', facecolor=(0,0,0,0), lw=2))

# --- 新增：模块化的绘图函数 ---
def generate_visualization(image_data, predictions, output_path, label_style='id', draw_masks=False, masks_data=None):
    plt.figure(figsize=(20, 20)); plt.imshow(image_data)
    if draw_masks and masks_data is not None and len(masks_data) > 0:
        for mask in masks_data: show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    
    for data in predictions:
        box = data['bbox']
        show_box(box, plt.gca())
        
        if label_style == 'id':
            label_text = f"ID:{data['id']}"
        elif label_style == 'confidence':
            label_text = f"{data['class_name']}: {data['confidence']:.2f}"
        else:
            label_text = ""
            
        plt.text(box[0], box[1] - 7, label_text, color='white', fontsize=10, bbox=dict(facecolor='green', alpha=0.7, pad=1))
        
    plt.axis('off'); plt.savefig(output_path, bbox_inches='tight', pad_inches=0); plt.close()

# --- 2. SAHI + YOLOv8 检测 ---
print(f"Stage 1 & 2: Running SAHI + YOLOv8 on '{args.filename}'...")
# ... (这部分代码和之前完全一样，为了简洁省略) ...
detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=YOLO_MODEL_PATH, confidence_threshold=0.35, device=DEVICE)
sahi_result = get_sliced_prediction(image=str(TEST_IMAGE_PATH), detection_model=detection_model, slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
predictions_data = [{'id': i + 1, 'class_name': p.category.name, 'confidence': p.score.value, 'bbox': p.bbox.to_xyxy()} for i, p in enumerate(sahi_result.object_prediction_list)]
print(f"Found {len(predictions_data)} objects.")


# --- 3. SAM 分割 ---
print("\nStage 3: Running SAM for segmentation...")
# ... (这部分代码和之前完全一样，为了简洁省略) ...
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
predictor = SamPredictor(sam)
image = cv2.cvtColor(cv2.imread(str(TEST_IMAGE_PATH)), cv2.COLOR_BGR2RGB)
predictor.set_image(image)
bboxes_for_sam = [data['bbox'] for data in predictions_data]
masks = []
if bboxes_for_sam:
    input_boxes = torch.tensor(bboxes_for_sam, device=predictor.device)
    masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes=input_boxes, multimask_output=False)
    print("SAM segmentation completed.")
else: print("No objects detected, skipping SAM.")


# --- 4. 生成所有输出文件 (V4版) ---
print("\nGenerating all output files with professional structure...")

# 4.1 复制原图
shutil.copy(TEST_IMAGE_PATH, ORIGINAL_DIR / f"{run_id}.jpg")

# 4.2 生成 "带掩码和置信度" 的结果图
confidence_img_path = CONFIDENCE_DIR / f"result_with_confidence{run_id}.png"
generate_visualization(image, predictions_data, confidence_img_path, label_style='confidence', draw_masks=True, masks_data=masks)
print(f"  - Saved confidence/mask result to: {confidence_img_path}")

# 4.3 生成 "只有框和ID" 的结果图
square_only_img_path = SQUARE_ONLY_DIR / f"result_only{run_id}.png"
generate_visualization(image, predictions_data, square_only_img_path, label_style='id', draw_masks=False)
print(f"  - Saved boxes-only result to: {square_only_img_path}")

# 4.4 记录图片来源路径
provenance_file = BASE_OUTPUT_DIR / 'picture_route.md'
if not provenance_file.exists():
    with open(provenance_file, 'w') as f: f.write("| Run ID | Original Filename |\n|:------:|:------------------|\n")
with open(provenance_file, 'a') as f: f.write(f"| {run_id} | `{args.filename}` |\n")
print(f"  - Updated provenance file: {provenance_file}")

# 4.5 导出 CSV (移动到 squareOnly 文件夹)
csv_path = SQUARE_ONLY_DIR / f"result_data_{run_id}.csv"
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'Class', 'Confidence', 'Bbox_XYXY'])
    for data in predictions_data: writer.writerow([data['id'], data['class_name'], f"{data['confidence']:.4f}", data['bbox']])
print(f"  - Saved structured data to: {csv_path}")

print("\nAll tasks completed successfully!")
