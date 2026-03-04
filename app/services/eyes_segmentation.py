"""
eyes_segmentation.py — SAM-based Eye Contour Segmentation

使用 SAM (Segment Anything Model) vit_b，以 keypoint 眼睛中心點
作為 point prompt，在動物 ROI 內精確分割每隻眼睛的輪廓 mask。
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from segment_anything import sam_model_registry, SamPredictor

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images" / "val2017"

# 輸入：keypoint detection 結果
KPT_RESULTS_JSON = DATA_DIR / "keypoints" / "keypoint_results.json"

# 輸出
EYE_SEG_DIR = DATA_DIR / "eyes_segmentation"
EYE_MASKS_DIR = EYE_SEG_DIR / "masks"
EYE_VIS_DIR = EYE_SEG_DIR / "visualizations"
EYE_SEG_RESULTS_JSON = EYE_SEG_DIR / "eye_segmentation_results.json"

# SAM 模型
MODEL_DIR = PROJECT_ROOT / "app" / "models"
SAM_TYPE = "vit_b"
SAM_CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CKPT_FILE = MODEL_DIR / "sam_vit_b_01ec64.pth"

# Eye box prompt 的 padding (像素)
EYE_BOX_PADDING = 30

# 視覺化配色
VIS_COLORS = [
    (1.0, 0.2, 0.2),   # 紅
    (0.2, 0.6, 1.0),   # 藍
    (0.2, 1.0, 0.4),   # 綠
    (1.0, 0.8, 0.1),   # 黃
    (0.8, 0.3, 1.0),   # 紫
    (1.0, 0.5, 0.1),   # 橘
    (0.1, 1.0, 0.9),   # 青
    (1.0, 0.4, 0.7),   # 粉
]


# ---------------------------------------------------------------------------
# 模型下載 & 載入
# ---------------------------------------------------------------------------

def download_checkpoint(url: str, dest: Path) -> None:
    """下載 SAM checkpoint。"""
    if dest.exists():
        print(f"  ✓ 權重已存在: {dest.name}")
        return

    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ 下載權重: {dest.name} (~375MB)")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r    {pct:5.1f}%  ({downloaded:,} / {total:,} bytes)", end="")
    print()


def load_sam_model(device: str = "cpu") -> SamPredictor:
    """
    載入 SAM vit_b 模型並建立 SamPredictor。

    Returns
    -------
    SamPredictor
    """
    print(f"\n🔧 載入 SAM ({SAM_TYPE}) ...")
    print(f"   Device: {device}")

    # 下載 checkpoint
    download_checkpoint(SAM_CKPT_URL, SAM_CKPT_FILE)

    # 載入模型
    sam = sam_model_registry[SAM_TYPE](checkpoint=str(SAM_CKPT_FILE))
    sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)
    print("   ✓ SAM 載入完成")

    return predictor


# ---------------------------------------------------------------------------
# Eye Segmentation
# ---------------------------------------------------------------------------

def make_eye_box(eye_point: dict, animal_bbox: list[float],
                 img_w: int, img_h: int,
                 padding: int = EYE_BOX_PADDING) -> np.ndarray:
    """
    以眼睛中心點建立小 box prompt，限制在動物 bbox 和圖片邊界內。

    Parameters
    ----------
    eye_point : {"x": float, "y": float}
    animal_bbox : [x1, y1, x2, y2]
    img_w, img_h : image dimensions
    padding : pixels to expand around eye center

    Returns
    -------
    np.ndarray [x1, y1, x2, y2]
    """
    cx, cy = eye_point["x"], eye_point["y"]
    ax1, ay1, ax2, ay2 = animal_bbox

    # Eye box
    ex1 = cx - padding
    ey1 = cy - padding
    ex2 = cx + padding
    ey2 = cy + padding

    # Clamp to animal bbox
    ex1 = max(ex1, ax1)
    ey1 = max(ey1, ay1)
    ex2 = min(ex2, ax2)
    ey2 = min(ey2, ay2)

    # Clamp to image boundary
    ex1 = max(ex1, 0)
    ey1 = max(ey1, 0)
    ex2 = min(ex2, img_w)
    ey2 = min(ey2, img_h)

    return np.array([ex1, ey1, ex2, ey2], dtype=np.float32)


def segment_single_eye(
    predictor: SamPredictor,
    eye_point: dict,
    animal_bbox: list[float],
    img_w: int,
    img_h: int,
    padding: int = EYE_BOX_PADDING,
) -> dict:
    """
    用 SAM 分割單隻眼睛。

    Parameters
    ----------
    predictor : SamPredictor (image already set)
    eye_point : {"x", "y", "confidence"}
    animal_bbox : [x1, y1, x2, y2]
    img_w, img_h : image dimensions

    Returns
    -------
    dict:
        mask: np.ndarray (H, W) bool
        mask_area_px: int
        score: float (SAM predicted IoU)
        eye_box: [x1, y1, x2, y2]
    """
    # Point prompt: eye center
    point_coords = np.array([[eye_point["x"], eye_point["y"]]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)  # 1 = foreground

    # Box prompt: small box around eye
    eye_box = make_eye_box(eye_point, animal_bbox, img_w, img_h, padding)

    # SAM predict
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=eye_box,
        multimask_output=True,  # 3 masks, pick best
    )

    # 取 score 最高的 mask
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx]  # (H, W) bool
    best_score = float(scores[best_idx])

    return {
        "mask": best_mask,
        "mask_area_px": int(best_mask.sum()),
        "score": best_score,
        "eye_box": eye_box.tolist(),
    }


# ---------------------------------------------------------------------------
# 儲存
# ---------------------------------------------------------------------------

def save_eye_mask(image_id: int, det_id: int, eye_side: str,
                  mask: np.ndarray) -> str:
    """儲存單隻眼睛的 mask，回傳檔名。"""
    EYE_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{image_id}_det{det_id}_{eye_side}.npz"
    np.savez_compressed(EYE_MASKS_DIR / fname, mask=mask)
    return fname


# ---------------------------------------------------------------------------
# 視覺化
# ---------------------------------------------------------------------------

def visualize_eye_segmentation(
    image_path: Path,
    animals: list[dict],
    save_path: Path,
) -> None:
    """
    視覺化：原圖 + 動物 bbox + 眼睛 mask overlay + 眼睛中心點。
    """
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)

    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(img_array)

    for idx, animal in enumerate(animals):
        color = VIS_COLORS[idx % len(VIS_COLORS)]
        bbox = animal["bbox"]
        x1, y1, x2, y2 = bbox

        # Animal bounding box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)

        # Label
        ax.text(
            x1, y1 - 6,
            f"{animal['label']} #{idx}",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=2),
        )

        # Eye masks & markers
        for eye_side in ["left_eye", "right_eye"]:
            eye_data = animal.get(eye_side)
            if not eye_data or not eye_data.get("mask_file"):
                continue

            # Load mask
            mask_path = EYE_MASKS_DIR / eye_data["mask_file"]
            if mask_path.exists():
                mask = np.load(mask_path)["mask"]
                # Overlay
                overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
                eye_color = (0.0, 1.0, 0.5) if eye_side == "left_eye" else (1.0, 0.5, 0.0)
                overlay[mask, :3] = eye_color
                overlay[mask, 3] = 0.5
                ax.imshow(overlay)

            # Center point
            center = eye_data["center"]
            marker = "o" if eye_side == "left_eye" else "s"
            ax.plot(center["x"], center["y"], marker, color="white",
                    markersize=8, markeredgecolor=color, markeredgewidth=2)

            # Eye box
            if "eye_box" in eye_data:
                eb = eye_data["eye_box"]
                rect = patches.Rectangle(
                    (eb[0], eb[1]), eb[2] - eb[0], eb[3] - eb[1],
                    linewidth=1, edgecolor="cyan", facecolor="none", linestyle=":",
                )
                ax.add_patch(rect)

            # Label
            side_label = "L" if eye_side == "left_eye" else "R"
            ax.text(
                center["x"] + 5, center["y"] - 8,
                f"{side_label} area={eye_data.get('mask_area_px', '?')}px²",
                fontsize=8, color="cyan", fontweight="bold",
            )

    ax.set_axis_off()
    ax.set_title(
        f"{image_path.stem}  —  {len(animals)} animal(s), eye segmentation",
        fontsize=13,
    )
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 批次處理
# ---------------------------------------------------------------------------

def process_all_images(
    kpt_results_json: Path = KPT_RESULTS_JSON,
    device: str = "cpu",
) -> list[dict]:
    """
    批次處理所有圖片：對每隻眼睛執行 SAM 分割。
    """
    # 載入 keypoint 結果
    print(f"\n📄 載入 keypoint 結果: {kpt_results_json.name}")
    with open(kpt_results_json, "r", encoding="utf-8") as f:
        kpt_results = json.load(f)
    print(f"   共 {len(kpt_results)} 張圖片")

    # 載入 SAM
    predictor = load_sam_model(device=device)

    all_results = []
    total = len(kpt_results)

    for i, entry in enumerate(kpt_results):
        image_id = entry["image_id"]
        file_name = entry["file_name"]
        image_path = IMAGES_DIR / file_name
        img_w = entry["width"]
        img_h = entry["height"]

        print(f"\n[{i+1}/{total}] {file_name} (image_id={image_id})")

        if not image_path.exists():
            print(f"   ⚠ 圖片不存在，跳過")
            continue

        # 載入圖片並設定 SAM embedding (一次)
        image = np.array(Image.open(image_path).convert("RGB"))
        print(f"   設定 image embedding ...", end="")
        predictor.set_image(image)
        print(" ✓")

        animals = []
        for animal in entry["animals"]:
            det_id = animal["detection_id"]
            label = animal["label"]
            bbox = animal["bbox"]

            print(f"   動物 #{det_id} ({label}): ", end="")

            animal_entry = {
                "detection_id": det_id,
                "label": label,
                "bbox": bbox,
                "score": animal["score"],
            }

            eyes_found = 0
            for eye_side in ["left_eye", "right_eye"]:
                eye_data = animal.get(eye_side)
                if eye_data is None:
                    animal_entry[eye_side] = None
                    continue

                # SAM segmentation
                result = segment_single_eye(
                    predictor, eye_data, bbox, img_w, img_h
                )

                # 儲存 mask
                mask_fname = save_eye_mask(image_id, det_id, eye_side, result["mask"])

                animal_entry[eye_side] = {
                    "center": {"x": eye_data["x"], "y": eye_data["y"]},
                    "confidence": eye_data["confidence"],
                    "mask_file": mask_fname,
                    "mask_area_px": result["mask_area_px"],
                    "sam_score": result["score"],
                    "eye_box": result["eye_box"],
                }
                eyes_found += 1

            side_info = []
            for s in ["left_eye", "right_eye"]:
                d = animal_entry.get(s)
                if d and d.get("mask_area_px"):
                    side_info.append(f"{s[0].upper()}={d['mask_area_px']}px²")
                else:
                    side_info.append(f"{s[0].upper()}=N/A")
            print(f"{', '.join(side_info)}")

            animals.append(animal_entry)

        # 視覺化
        vis_path = EYE_VIS_DIR / f"{image_id}_eye_seg.jpg"
        visualize_eye_segmentation(image_path, animals, vis_path)
        print(f"   視覺化: {vis_path.name}")

        all_results.append({
            "image_id": image_id,
            "file_name": file_name,
            "width": img_w,
            "height": img_h,
            "animals": animals,
        })

    # 儲存結果 JSON
    EYE_SEG_DIR.mkdir(parents=True, exist_ok=True)
    with open(EYE_SEG_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 結果已儲存: {EYE_SEG_RESULTS_JSON}")

    # 摘要
    total_animals = sum(len(r["animals"]) for r in all_results)
    total_eyes = sum(
        1 for r in all_results
        for a in r["animals"]
        for s in ["left_eye", "right_eye"]
        if a.get(s) and a[s].get("mask_file")
    )
    print(f"\n{'='*60}")
    print(f" ✅ Eye Segmentation 完成")
    print(f"    處理圖片: {len(all_results)}")
    print(f"    動物數: {total_animals}")
    print(f"    眼睛 mask 數: {total_eyes}")
    print(f"    結果: {EYE_SEG_RESULTS_JSON}")
    print(f"    視覺化: {EYE_VIS_DIR}")
    print(f"{'='*60}")

    return all_results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = process_all_images(device="cpu")
