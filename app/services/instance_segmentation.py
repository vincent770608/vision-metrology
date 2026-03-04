"""
instance_segmentation.py — Mask R-CNN 動物個體分割

使用 torchvision 預訓練的 Mask R-CNN (maskrcnn_resnet50_fpn_v2) 對篩選後的
COCO 圖片執行 Instance Segmentation，產出每隻動物的 binary mask。
後續眼睛偵測將限制在各 mask 的 ROI 內，降低誤檢。
"""

import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FILTERED_JSON = DATA_DIR / "filtered" / "filtered_animal_images.json"
IMAGES_DIR = DATA_DIR / "images" / "val2017"

# 輸出目錄
SEG_DIR = DATA_DIR / "segmentation"
MASKS_DIR = SEG_DIR / "masks"
VIS_DIR = SEG_DIR / "visualizations"

# COCO 類別對照 —— torchvision Mask R-CNN 的 label index（1-indexed）
# 參考: torchvision.models.detection.coco_utils
# 0 = __background__
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

# 動物類別名稱（與 COCO supercategory == 'animal' 對應）
ANIMAL_NAMES = {
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
}

# 動物類別的 label index（在 COCO_INSTANCE_CATEGORY_NAMES 中的位置）
ANIMAL_LABEL_IDS = {
    i for i, name in enumerate(COCO_INSTANCE_CATEGORY_NAMES)
    if name in ANIMAL_NAMES
}

# 信心分數閾值
DEFAULT_SCORE_THRESHOLD = 0.7

# mask 二值化閾值
MASK_THRESHOLD = 0.5

# 視覺化配色（最多 10 隻動物，循環使用）
VIS_COLORS = [
    (1.0, 0.2, 0.2, 0.45),   # 紅
    (0.2, 0.6, 1.0, 0.45),   # 藍
    (0.2, 1.0, 0.4, 0.45),   # 綠
    (1.0, 0.8, 0.1, 0.45),   # 黃
    (0.8, 0.3, 1.0, 0.45),   # 紫
    (1.0, 0.5, 0.1, 0.45),   # 橘
    (0.1, 1.0, 0.9, 0.45),   # 青
    (1.0, 0.4, 0.7, 0.45),   # 粉
    (0.5, 0.8, 0.2, 0.45),   # 黃綠
    (0.6, 0.4, 0.2, 0.45),   # 棕
]


# ---------------------------------------------------------------------------
# 模型載入
# ---------------------------------------------------------------------------

def load_model(device: str = "auto") -> tuple:
    """
    載入預訓練 Mask R-CNN 模型。

    Parameters
    ----------
    device : str
        "auto" / "cuda" / "cpu"

    Returns
    -------
    (model, device)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(f"\n🔧 載入 Mask R-CNN (maskrcnn_resnet50_fpn_v2) ...")
    print(f"   Device: {device}")

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)
    model.to(device)
    model.eval()

    print("   ✓ 模型載入完成")
    return model, device


# ---------------------------------------------------------------------------
# 推論 & 篩選
# ---------------------------------------------------------------------------

def run_inference(model, image_path: Path, device: torch.device) -> dict:
    """
    對單張圖片執行 Mask R-CNN 推論。

    Returns
    -------
    dict with keys: boxes, labels, scores, masks
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        predictions = model([image_tensor])[0]

    # 將結果移回 CPU
    return {k: v.cpu() for k, v in predictions.items()}


def filter_animal_detections(
    predictions: dict,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[dict]:
    """
    從推論結果中篩選動物類別、信心 ≥ 閾值的偵測。

    Returns
    -------
    list of dict, 每筆包含:
        label (str), label_id (int), score (float),
        bbox (list[float]), mask (np.ndarray H×W bool)
    """
    detections = []
    labels = predictions["labels"].numpy()
    scores = predictions["scores"].numpy()
    boxes = predictions["boxes"].numpy()
    masks = predictions["masks"].numpy()  # (N, 1, H, W)

    for i in range(len(labels)):
        label_id = int(labels[i])
        score = float(scores[i])

        if label_id not in ANIMAL_LABEL_IDS:
            continue
        if score < score_threshold:
            continue

        # 二值化 mask: (1, H, W) → (H, W) bool
        binary_mask = (masks[i, 0] >= MASK_THRESHOLD)

        detections.append({
            "label": COCO_INSTANCE_CATEGORY_NAMES[label_id],
            "label_id": label_id,
            "score": score,
            "bbox": boxes[i].tolist(),  # [x1, y1, x2, y2]
            "mask": binary_mask,        # np.ndarray (H, W) bool
        })

    return detections


def segment_animals_in_image(
    model,
    image_path: Path,
    device: torch.device,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[dict]:
    """
    對單張圖片執行推論 + 篩選，回傳動物偵測結果。
    """
    predictions = run_inference(model, image_path, device)
    detections = filter_animal_detections(predictions, score_threshold)
    return detections


# ---------------------------------------------------------------------------
# 儲存結果
# ---------------------------------------------------------------------------

def save_masks(image_id: int, detections: list[dict]) -> list[str]:
    """
    將每隻動物的 binary mask 儲存為 .npz 檔。

    Returns
    -------
    list of mask filenames
    """
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    mask_files = []

    for idx, det in enumerate(detections):
        fname = f"{image_id}_det{idx}.npz"
        np.savez_compressed(MASKS_DIR / fname, mask=det["mask"])
        mask_files.append(fname)

    return mask_files


def save_results(all_results: list[dict]) -> Path:
    """
    將所有圖片的偵測 metadata 存為 JSON。

    Parameters
    ----------
    all_results : list of dict
        每筆包含 image_id, file_name, detections (不含 mask ndarray)
    """
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    out_file = SEG_DIR / "segmentation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 結果已儲存: {out_file}")
    return out_file


# ---------------------------------------------------------------------------
# 視覺化
# ---------------------------------------------------------------------------

def visualize_segmentation(
    image_path: Path,
    detections: list[dict],
    save_path: Path,
) -> None:
    """
    產出視覺化圖片：半透明 mask overlay + bounding box + label。
    """
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)

    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(img_array)

    for idx, det in enumerate(detections):
        color = VIS_COLORS[idx % len(VIS_COLORS)]
        rgb = color[:3]
        alpha = color[3]
        mask = det["mask"]

        # 半透明 mask overlay
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask, :3] = rgb
        overlay[mask, 3] = alpha
        ax.imshow(overlay)

        # Bounding box
        x1, y1, x2, y2 = det["bbox"]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2,
            edgecolor=rgb,
            facecolor="none",
        )
        ax.add_patch(rect)

        # Label + score
        label_text = f"{det['label']} {det['score']:.2f}"
        ax.text(
            x1, y1 - 6,
            label_text,
            fontsize=11,
            fontweight="bold",
            color="white",
            bbox=dict(facecolor=rgb, alpha=0.8, edgecolor="none", pad=2),
        )

    ax.set_axis_off()
    ax.set_title(f"{image_path.stem}  —  {len(detections)} animal(s) detected", fontsize=13)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 批次處理
# ---------------------------------------------------------------------------

def process_all_images(
    filtered_json: Path = FILTERED_JSON,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[dict]:
    """
    批次處理所有篩選過的圖片：推論 → 篩選 → 存 mask → 視覺化。

    Returns
    -------
    list of metadata dicts（已排除 mask ndarray，適合 JSON 序列化）
    """
    # 載入篩選結果
    print(f"\n📄 載入篩選結果: {filtered_json.name}")
    with open(filtered_json, "r", encoding="utf-8") as f:
        filtered = json.load(f)
    print(f"   共 {len(filtered)} 張圖片")

    # 載入模型
    model, device = load_model()

    all_results = []
    total = len(filtered)

    for i in range(total):
        entry = filtered[i]
        image_id = entry["image_id"]
        file_name = entry["file_name"]
        image_path = IMAGES_DIR / file_name

        print(f"\n[{i+1}/{total}] {file_name} (image_id={image_id})")

        # 檢查圖片是否存在
        if not image_path.exists():
            print(f"   ⚠ 圖片不存在，跳過: {image_path}")
            continue

        # 推論 + 篩選
        detections = segment_animals_in_image(model, image_path, device, score_threshold)
        print(f"   偵測到 {len(detections)} 隻動物: "
              f"{[d['label'] for d in detections]}")

        # 儲存 masks
        mask_files = save_masks(image_id, detections)

        # 視覺化
        vis_path = VIS_DIR / f"{image_id}_seg.jpg"
        visualize_segmentation(image_path, detections, vis_path)
        print(f"   視覺化: {vis_path.name}")

        # 組裝 metadata（不含 numpy mask）
        result_entry = {
            "image_id": image_id,
            "file_name": file_name,
            "width": entry["width"],
            "height": entry["height"],
            "detections": [
                {
                    "detection_id": idx,
                    "label": det["label"],
                    "label_id": det["label_id"],
                    "score": det["score"],
                    "bbox": det["bbox"],
                    "mask_file": mask_files[idx],
                }
                for idx, det in enumerate(detections)
            ],
        }
        all_results.append(result_entry)

    # 儲存所有結果
    save_results(all_results)

    # 摘要
    total_detections = sum(len(r["detections"]) for r in all_results)
    print(f"\n{'='*60}")
    print(f" ✅ Instance Segmentation 完成")
    print(f"    處理圖片: {len(all_results)}")
    print(f"    偵測動物: {total_detections}")
    print(f"    Masks:    {MASKS_DIR}")
    print(f"    視覺化:   {VIS_DIR}")
    print(f"{'='*60}")

    return all_results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = process_all_images()
