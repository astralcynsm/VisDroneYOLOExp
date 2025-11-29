import cv2
import os # 解决不知道哪里的死锁用的lmfao
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
cv2.setNumThreads(0)
import sys # same as above
import psutil # same
import torch
import numpy as np
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import argparse
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
import time

# --- -1. 资源监控 ---
def check_memory_and_exit(threshold_gb=5):
    try:
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss
        mem_gb = mem_bytes / (1024 ** 3)

        print(f"Final memory check: Usage = {mem_gb:.2f} GB")

        if mem_gb > threshold_gb:
            print(f"WARNING: Memory usage is exploding, exceeding threshold of {threshold_gb} GB")
            print(f"This indicates a potential memory leak. Forcing exit to prevent system instability. ")
            os._exit(0) # Force exit

        else:
            print("Memory usage is within limits. Exiting.")
            sys.exit(0)
    except Exception as e:
        print(f"An error occurred during final memory check: {e}")
        print(f"Forcing exit as a fallback.")
        os._exit(1)
# --- 0. 命令行参数 ---
parser = argparse.ArgumentParser(description="V4.3: Final, Stable, Presentation-Ready Comparator.")
parser.add_argument('--filename', type=str, required=True, help='Filename of the image to process.')
args = parser.parse_args()

# --- 1. 配置与路径 ---
BASE_IMAGE_PATH = Path('/mnt/Storage/files/MachineLearning/VisDroneDataset/Datasets/VisDrone2019-DET-train/images/')
YOLO_MODEL_PATH = '/mnt/Storage/files/MachineLearning/VisDroneDataset/runs/from_colab/detect/visdrone_yolov8l_A100_colab/weights/best.pt'
SAM_CHECKPOINT_PATH = '/mnt/Storage/files/MachineLearning/VisDroneDataset/SAHIandSAM/sam_vit_b_01ec64.pth'
SAM_MODEL_TYPE = "vit_b"
TEST_IMAGE_PATH = BASE_IMAGE_PATH / args.filename
IOU_THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not TEST_IMAGE_PATH.is_file():
    print(f"Error: Image file not found at '{TEST_IMAGE_PATH}'")
    exit()

# --- 2. 辅助函数 ---
def calculate_iou(box1, box2):
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
    detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=model_path, confidence_threshold=0.35, device=DEVICE)
    sahi_result = get_sliced_prediction(image=str(image_path), detection_model=detection_model, slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
    return [{'class_name': p.category.name, 'confidence': p.score.value, 'bbox': p.bbox.to_xyxy()} for p in sahi_result.object_prediction_list]

def run_standard_prediction(model, image_path):
    results = model.predict(source=str(image_path), conf=0.35, verbose=False)
    boxes = results[0].boxes
    return [{'class_name': model.names[int(b.cls)], 'confidence': float(b.conf), 'bbox': b.xyxy[0].cpu().numpy().tolist()} for b in boxes]

def draw_results(image, predictions, masks=None, legend_items=None):
    plt.figure(figsize=(20, 20), dpi=150)
    plt.imshow(image)
    ax = plt.gca()
    
    if masks is not None and len(masks) > 0:
        for mask in masks:
            mask_cpu = mask.cpu().numpy()
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            h, w = mask_cpu.shape[-2:]
            mask_image = mask_cpu.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)
            
    for pred in predictions:
        box = pred['bbox']
        label = f"{pred['class_name']}: {pred['confidence']:.2f}"
        color_rgb = tuple(c/255 for c in pred['color'])[::-1]
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor=color_rgb, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(box[0], box[1] - 7, label, color='white', fontsize=10, bbox=dict(facecolor=color_rgb, alpha=0.8, pad=1))
    
    if legend_items is not None:
        y_pos = 60
        for text, color_bgr in legend_items.items():
            color_rgb = tuple(c/255 for c in color_bgr)[::-1]
            ax.add_patch(plt.Rectangle((30, y_pos - 25), 30, 30, facecolor=color_rgb, edgecolor='white', lw=1))
            ax.text(75, y_pos, text, color='white', fontsize=20, va='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=4))
            y_pos += 50

    ax.axis('off')
    return plt

# --- 3. 主流程 ---
with torch.no_grad():
    yolo_model = YOLO(YOLO_MODEL_PATH)

    print("Running SAHI prediction..."); sahi_preds = run_sahi_prediction(YOLO_MODEL_PATH, TEST_IMAGE_PATH); print(f"  -> Found {len(sahi_preds)} objects.")
    print("Running Standard YOLO prediction..."); standard_preds = run_standard_prediction(yolo_model, TEST_IMAGE_PATH); print(f"  -> Found {len(standard_preds)} objects.")

    print("Matching predictions using IoU...");
    matches, sahi_indices_matched, standard_indices_matched = [], set(), set()
    for i, sahi_pred in enumerate(sahi_preds):
        best_iou, best_j = 0, -1
        for j, std_pred in enumerate(standard_preds):
            if j in standard_indices_matched:
                continue
            iou = calculate_iou(sahi_pred['bbox'], std_pred['bbox'])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou > IOU_THRESHOLD:
            matches.append({'sahi': sahi_pred, 'standard': standard_preds[best_j], 'iou': best_iou})
            sahi_indices_matched.add(i); standard_indices_matched.add(best_j)

    sahi_only = [sahi_preds[i] for i in range(len(sahi_preds)) if i not in sahi_indices_matched]
    standard_only = [standard_preds[i] for i in range(len(standard_preds)) if i not in standard_indices_matched]
    print(f"  -> Matched: {len(matches)}, SAHI Only: {len(sahi_only)}, Standard Only: {len(standard_only)}")

    print("Running SAM on SAHI predictions...");
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
    predictor = SamPredictor(sam)
    image = cv2.cvtColor(cv2.imread(str(TEST_IMAGE_PATH)), cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    bboxes_for_sam = [p['bbox'] for p in sahi_preds]
    sahi_masks = []
    if bboxes_for_sam:
        masks_tensor, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=torch.tensor(bboxes_for_sam, device=predictor.device),
            multimask_output=False
        )
        sahi_masks = [m[0] for m in masks_tensor]
    print("  -> SAM segmentation completed.")

    del sam
    del predictor
    del masks_tensor
    gc.collect()
    torch.cuda.empty_cache()

# --- 4. 生成输出 ---
run_id = Path(args.filename).stem
output_dir = Path(f'comparison_results/{run_id}')
output_dir.mkdir(parents=True, exist_ok=True)

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

COLOR_MATCHED = (0, 255, 0) # Green (BGR)
COLOR_SAHI_ONLY = (255, 0, 0) # Blue (BGR)
COLOR_YOLO_ONLY = (0, 0, 255) # Red (BGR)
legend_std = {"Matched": COLOR_MATCHED, "Standard YOLO Only": COLOR_YOLO_ONLY}
legend_sahi = {"Matched": COLOR_MATCHED, "SAHI Only": COLOR_SAHI_ONLY}

sahi_view_preds = [{**match['sahi'], 'color': COLOR_MATCHED} for match in matches] + [{**pred, 'color': COLOR_SAHI_ONLY} for pred in sahi_only]
standard_view_preds = [{**match['standard'], 'color': COLOR_MATCHED} for match in matches] + [{**pred, 'color': COLOR_YOLO_ONLY} for pred in standard_only]

plt_std = draw_results(image.copy(), standard_view_preds, legend_items=legend_std)
plt_std.savefig(output_dir / 'comparison_Standard_view.png', bbox_inches='tight', pad_inches=0)
plt.close()

plt_sahi = draw_results(image.copy(), sahi_view_preds, legend_items=legend_sahi)
plt_sahi.savefig(output_dir / 'comparison_SAHI_view.png', bbox_inches='tight', pad_inches=0)
plt.close()

plt_sahi_sam = draw_results(image.copy(), sahi_view_preds, masks=sahi_masks, legend_items=legend_sahi)
plt_sahi_sam.savefig(output_dir / 'comparison_SAHI_with_SAM.png', bbox_inches='tight', pad_inches=0)
plt.close()

print(f"  -> Visual comparison images with legends saved to: {output_dir}")

time.sleep(5)

check_memory_and_exit(threshold_gb=5)
