import os
import shutil
from tqdm import tqdm
from PIL import Image
import yaml

visdrone_train_path = '/mnt/Storage/files/MachineLearning/VisDroneDataset/Datasets/VisDrone2019-DET-train/'
visdrone_val_path = '/mnt/Storage/files/MachineLearning/VisDroneDataset/Datasets/VisDrone2019-DET-val/'
# visdrone_test_challenge_path = '/home/cynsm/VisDroneDataset/Datasets/VisDrone2019-DET-test-challenge/'
visdrone_test_dev_path = '/mnt/Storage/files/MachineLearning/VisDroneDataset/Datasets/VisDrone2019-DET-test-dev/'

output_path = './VisDrone-YOLO' 

# VisDrone official category: pedestrian (1), people (2), bicycle (3), car (4), van (5), 
# truck (6), tricycle (7), awning-tricycle (8), bus (9), motor (10)
# categories we need: person, vehicle, motorcycle


# define a mapping dictionary
# key is VisDrone's category ID，string
# value is the new category ID, int
CLASS_MAPPING = {
    '1': 0,  # pedestrian -> person
    '2': 0,  # people -> person
    # '3': 2, # bicycle -> motorcycle
    '4': 1,  # car -> vehicle
    '5': 1,  # van -> vehicle
    '6': 1,  # truck -> vehicle
    '9': 1,  # bus -> vehicle
    '10': 2, # motor -> motorcycle
}

# 我们最终的类别名字
class_names = ['person', 'vehicle', 'motorcycle']

def convert_annotations(source_path, dest_path_type):
    """
    转换标注文件并整理目录结构
    :param source_path: VisDrone的源目录 (e.g., VisDrone2019-DET-train)
    :param dest_path_type: 'train' or 'valid'
    """
    source_images_path = os.path.join(source_path, 'images')
    source_annotations_path = os.path.join(source_path, 'annotations')

    dest_images_path = os.path.join(output_path, dest_path_type, 'images')
    dest_labels_path = os.path.join(output_path, dest_path_type, 'labels')

    # 创建目标文件夹
    os.makedirs(dest_images_path, exist_ok=True)
    os.makedirs(dest_labels_path, exist_ok=True)

    # 获取所有标注文件的列表
    annotation_files = os.listdir(source_annotations_path)
    print(f"\nProcessing {dest_path_type} set...")

    for filename in tqdm(annotation_files, desc=f"Converting {dest_path_type}"):
        # 获取文件名（不带后缀）
        base_name = os.path.splitext(filename)[0]
        
        # 定义源文件和目标文件的完整路径
        source_ann_file = os.path.join(source_annotations_path, filename)
        source_img_file = os.path.join(source_images_path, f"{base_name}.jpg")
        
        dest_img_file = os.path.join(dest_images_path, f"{base_name}.jpg")
        dest_label_file = os.path.join(dest_labels_path, f"{base_name}.txt")

        # 1. 复制图片文件
        if os.path.exists(source_img_file):
            shutil.copy(source_img_file, dest_img_file)
        else:
            print(f"Warning: Image file not found for annotation {filename}, skipping.")
            continue

        # 2. 获取图片尺寸用于归一化
        with Image.open(source_img_file) as img:
            img_width, img_height = img.size

        # 3. 读取并转换标注
        yolo_annotations = []
        with open(source_ann_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                # VisDrone格式: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,...
                
                visdrone_class_id = parts[5]

                # 如果这个类别是我们需要的，就处理它
                if visdrone_class_id in CLASS_MAPPING:
                    new_class_id = CLASS_MAPPING[visdrone_class_id]
                    
                    # 读取像素坐标
                    bbox_left = float(parts[0])
                    bbox_top = float(parts[1])
                    bbox_width = float(parts[2])
                    bbox_height = float(parts[3])

                    # 转换成YOLO格式 (中心点x, 中心点y, 宽度, 高度)
                    x_center = bbox_left + bbox_width / 2
                    y_center = bbox_top + bbox_height / 2

                    # 归一化
                    x_center_norm = x_center / img_width
                    y_center_norm = y_center / img_height
                    width_norm = bbox_width / img_width
                    height_norm = bbox_height / img_height
                    
                    yolo_annotations.append(f"{new_class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}")

        # 4. 写入新的YOLO格式标注文件
        if yolo_annotations:
            with open(dest_label_file, 'w') as f:
                f.write("\n".join(yolo_annotations))

def main():
    # 处理训练集
    convert_annotations(visdrone_train_path, 'train')
    
    # 处理验证集
    convert_annotations(visdrone_val_path, 'valid')

#     convert_annotations(visdrone_test_challenge_path, 'challenge')

    convert_annotations(visdrone_test_dev_path, 'test')
    
    # --- 3. 创建 data.yaml 文件 ---
    data_yaml = {
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = os.path.join(output_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
        
    print(f"\nConversion complete! YOLO dataset is ready at: {output_path}")
    print(f"Configuration file 'data.yaml' created at: {yaml_path}")
    print("\nYou are now ready to train your model!")

if __name__ == '__main__':
    main()
