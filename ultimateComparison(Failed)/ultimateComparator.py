import cv2
import torch
import numpy as np
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import matplotlib.pyplot as plt
import argparse
import csv
import shutil

# --- 0. 命令行参数 ---
parser = argparse.ArgumentParser(description="V5: Intelligent pipeline with SAM-based verification.")
parser.add_argument('--filename', type=str, required=True, help='Filename of the image to process.')
args = parser.parse_args()

# --- 1. 配置与路径 ---
BASE_IMAGE_PATH = Path('/mnt/Storage/files/MachineLearning/VisDroneDataset/Datasets/VisDrone2019-DET-train/images/')
YOLO_MODEL_PATH = '/mnt/Storage/files/MachineLearning/VisDroneDataset/runs/from_colab/detect/visdrone_yolov8l_A100_colab/weights/best.pt'
SAM_CHECKPOINT_PATH = '/mnt/Storage/files/MachineLearning/VisDroneDataset/SAHIandSAM/sam_vit_b_01ec64.pth'
SAM_MODEL_TYPE = "vit_b"
TEST_IMAGE_PATH = BASE_IMAGE_PATH / args.filename
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not TEST_IMAGE_PATH.is_file(): exit(f"Error: Image file not found at '{TEST_IMAGE_PATH}'")

# --- V5 新增：合理性验证的阈值 ---
MAX_CONTOUR_COUNT = 5  # 最大轮廓数，超过即认为“太碎”
MIN_MASK_BOX_RATIO = 0.3 # 最小掩码-框面积比，低于即认为“太空”

# --- 结构化输出 ---
BASE_OUTPUT_DIR = Path('results_v5/')
FINAL_DIR = BASE_OUTPUT_DIR / 'final_verified'
FILTERED_DIR = BASE_OUTPUT_DIR / 'filtered_out'
ORIGINAL_DIR = BASE_OUTPUT_DIR / 'original'
for d in [FINAL_DIR, FILTERED_DIR, ORIGINAL_DIR]: d.mkdir(parents=True, exist_ok=True)
run_id = max([int(p.stem) for p in ORIGINAL_DIR.glob('*.jpg')] + [0]) + 1
print(f"Assigning new Run ID: {run_id}")

# --- 辅助函数 ---
# ... (show_mask, show_box 省略，与v4相同) ...
def show_mask(mask, ax, random_color=False):
    if random_color: color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else: color = np.array([30/255, 144/255, 255/255, 0.6]); h, w = mask.shape[-2:]; mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1); ax.imshow(mask_image)
def show_box(box, ax, color='lime'):
    x0, y0 = box[0], box[1]; w, h = box[2] - x0, box[3] - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))

# --- 2. SAHI + YOLOv8 检测 (与v4相同) ---
print(f"Stage 1 & 2: Running SAHI + YOLOv8 on '{args.filename}'...")
detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=YOLO_MODEL_PATH, confidence_threshold=0.35, device=DEVICE)
sahi_result = get_sliced_prediction(image=str(TEST_IMAGE_PATH), detection_model=detection_model, slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
candidate_predictions = [{'id': i + 1, 'class_name': p.category.name, 'confidence': p.score.value, 'bbox': p.bbox.to_xyxy()} for i, p in enumerate(sahi_result.object_prediction_list)]
print(f"Found {len(candidate_predictions)} candidate objects.")

# --- 3. SAM 分割 & 验证 (V5 核心) ---
print("\nStage 3: Running SAM for segmentation and verification...")
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
predictor = SamPredictor(sam)
image = cv2.cvtColor(cv2.imread(str(TEST_IMAGE_PATH)), cv2.COLOR_BGR2RGB)
predictor.set_image(image)

verified_predictions = []
filtered_predictions = []

# 逐个处理候选框
for pred in candidate_predictions:
    input_box = torch.tensor([pred['bbox']], device=predictor.device)
    masks, scores, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes=input_box, multimask_output=False)
    
    # 获取唯一的掩码
    mask = masks[0, 0].cpu().numpy() # (H, W) boolean mask
    
    # --- 合理性验证 ---
    is_verified = True
    reason = "Verified"
    
    # 1. 分块度验证
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > MAX_CONTOUR_COUNT:
        is_verified = False
        reason = f"Filtered: Too fragmented (contours={len(contours)} > {MAX_CONTOUR_COUNT})"
    
    # 2. 面积比验证
    if is_verified:
        mask_area = np.sum(mask)
        box = pred['bbox']
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        ratio = mask_area / box_area if box_area > 0 else 0
        if ratio < MIN_MASK_BOX_RATIO:
            is_verified = False
            reason = f"Filtered: Mask-Box ratio too low ({ratio:.2f} < {MIN_MASK_BOX_RATIO})"
            
    # 根据验证结果进行分类
    pred['verification_reason'] = reason
    if is_verified:
        verified_predictions.append(pred)
        pred['mask'] = masks[0,0] # 保存验证通过的掩码
    else:
        filtered_predictions.append(pred)

print(f"Verification complete: {len(verified_predictions)} verified, {len(filtered_predictions)} filtered.")

# --- 4. 生成所有输出文件 (V5版) ---
print("\nGenerating V5 output files...")
shutil.copy(TEST_IMAGE_PATH, ORIGINAL_DIR / f"{run_id}.jpg")

# 4.1 生成最终验证通过的结果图
plt.figure(figsize=(20, 20)); plt.imshow(image)
for pred in verified_predictions:
    show_mask(pred['mask'].cpu().numpy(), plt.gca(), random_color=True)
    show_box(pred['bbox'], plt.gca(), color='lime') # 绿色框代表通过
    label = f"{pred['class_name']}: {pred['confidence']:.2f}"
    plt.text(pred['bbox'][0], pred['bbox'][1] - 7, label, color='white', fontsize=10, bbox=dict(facecolor='green', alpha=0.8, pad=1))
plt.axis('off'); plt.savefig(FINAL_DIR / f"final_verified_{run_id}.png", bbox_inches='tight', pad_inches=0); plt.close()
print(f"  - Saved final verified result to: {FINAL_DIR}")

# 4.2 生成被过滤掉的误报图 (用于分析)
plt.figure(figsize=(20, 20)); plt.imshow(image)
for pred in filtered_predictions:
    show_box(pred['bbox'], plt.gca(), color='red') # 红色框代表被过滤
    label = f"{pred['class_name']}: {pred['confidence']:.2f} ({pred['verification_reason'].split(':')[1].strip()})"
    plt.text(pred['bbox'][0], pred['bbox'][1] - 7, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.8, pad=1))
plt.axis('off'); plt.savefig(FILTERED_DIR / f"filtered_out_{run_id}.png", bbox_inches='tight', pad_inches=0); plt.close()
print(f"  - Saved filtered-out result to: {FILTERED_DIR}")

# 4.3 导出包含验证信息的最终CSV报告
csv_path = BASE_OUTPUT_DIR / f"verification_report_{run_id}.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'Class', 'Confidence', 'Bbox_XYXY', 'Status', 'Reason'])
    for pred in verified_predictions + filtered_predictions:
        status = "Verified" if pred['verification_reason'] == "Verified" else "Filtered"
        writer.writerow([pred['id'], pred['class_name'], f"{pred['confidence']:.4f}", pred['bbox'], status, pred['verification_reason']])
print(f"  - Saved full verification report to: {csv_path}")

print("\nAll V5 tasks completed successfully!")
