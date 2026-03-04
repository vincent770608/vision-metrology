"""
config.py — 集中設定

所有路徑、模型 URL、閾值等參數在此統一管理。
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 路徑
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images" / "val2017"
MODELS_DIR = PROJECT_ROOT / "app" / "models"

# ---------------------------------------------------------------------------
# Mask R-CNN (torchvision)
# ---------------------------------------------------------------------------
MASKRCNN_SCORE_THRESHOLD = 0.7
MASK_THRESHOLD = 0.5

# COCO 類別（torchvision Mask R-CNN 的 label index，1-indexed）
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "N/A", "hat", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

ANIMAL_NAMES = {
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
}

ANIMAL_LABEL_IDS = {
    i for i, name in enumerate(COCO_INSTANCE_CATEGORY_NAMES)
    if name in ANIMAL_NAMES
}

# ---------------------------------------------------------------------------
# HRNet-w32 AP-10K (ONNX)
# ---------------------------------------------------------------------------
HRNET_CKPT_URL = (
    "https://download.openmmlab.com/mmpose/animal/hrnet/"
    "hrnet_w32_ap10k_256x256-18aac840_20211029.pth"
)
HRNET_CKPT_FILE = MODELS_DIR / "hrnet_w32_ap10k_256x256-18aac840_20211029.pth"
HRNET_ONNX_FILE = MODELS_DIR / "hrnet_w32_ap10k.onnx"

HRNET_INPUT_SIZE = (256, 256)       # (H, W)
HRNET_HEATMAP_SIZE = (64, 64)       # (H, W)
NUM_KEYPOINTS = 17
LEFT_EYE_IDX = 0
RIGHT_EYE_IDX = 1
KPT_SCORE_THRESHOLD = 0.3

# ImageNet normalization
PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# SAM (Segment Anything)
# ---------------------------------------------------------------------------
SAM_TYPE = "vit_b"
SAM_CKPT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)
SAM_CKPT_FILE = MODELS_DIR / "sam_vit_b_01ec64.pth"
EYE_BOX_PADDING = 30

# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------
MIN_BBOX_AREA = 3000    # 跳過太小的動物
