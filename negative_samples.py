import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from functools import reduce
import operator

# --- 配置 ---
TARGET_CLASSES = [
    "/m/01g317",  # Person
    "/m/0k4j",    # Car
    "/m/0199g",   # Bicycle
    "/m/01bjv",   # Bus
    "/m/04_sv",   # Motorcycle
    "/m/07jdr",   # Truck
]
NUM_SAMPLES = 2000
OUTPUT_DIR = "/mnt/Storage/files/MachineLearning/VisDroneDataset/oid_negatives"

# --- 主体 ---
print("正在加载 Open Images 数据集元信息...")
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["classifications"],
    max_samples=1,
)

print("正在构建查询视图...")
# 注意：这里我们不再需要 select_fields，因为 match 会自动处理
view = dataset.view()

# ==========================================================
#vvvvvvvvvvv 这是最终的、绝对可靠的解决方案 vvvvvvvvvvv
#
print("正在构建复杂的筛选条件...")
# 为我们的每一个目标类别，创建一个单独的“包含”条件
conditions = [
    F("classifications.label").contains(c) for c in TARGET_CLASSES
]

# 使用 functools.reduce 和 operator.and_，将所有条件用“与”(&)连接起来
combined_condition = reduce(operator.and_, conditions)

print("正在应用筛选条件...")
# 将这个复合条件应用到数据集上
view = dataset.match(combined_condition)
#
#^^^^^^^^^^^ 这是最终的、绝对可靠的解决方案 ^^^^^^^^^^^
#==========================================================


print("正在统计符合条件的样本总数 (这可能需要一些时间)...")
total_matching = view.count()
print(f"总共找到 {total_matching} 个符合条件的负样本。")

if total_matching == 0:
    print("错误：没有找到任何符合所有条件的负样本。请检查你的 TARGET_CLASSES 列表是否正确。")
    exit()
elif total_matching < NUM_SAMPLES:
    print(f"警告：找到的样本数({total_matching})少于你期望的数量({NUM_SAMPLES})。将使用所有找到的样本。")


print(f"正在从符合条件的样本中随机抽取 {NUM_SAMPLES} 个...")
negative_view = view.take(NUM_SAMPLES, seed=51)

print("开始下载图片...")
negative_view.export(
    export_dir=OUTPUT_DIR,
    dataset_type=fo.types.ImageDirectory,
)

print(f"下载完成！{NUM_SAMPLES} 张高质量负样本已保存到: {OUTPUT_DIR}")
