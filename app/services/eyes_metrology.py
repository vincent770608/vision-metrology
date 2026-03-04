"""
eyes_metrology.py — 眼距量測

量測項目:
1. 同一隻動物的雙眼距離（intra-animal）
   - 優先使用 keypoint 模型的雙眼中心點
   - Fallback: 眼睛 mask 的幾何中心 (centroid)
2. 任意兩隻動物的右眼距離（inter-animal pairwise）
   - 同圖中取所有動物 pair，計算兩隻的右眼中心點距離
"""

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images" / "val2017"

# 輸入：eye segmentation 結果
EYE_SEG_RESULTS_JSON = DATA_DIR / "eyes_segmentation" / "eye_segmentation_results.json"
EYE_MASKS_DIR = DATA_DIR / "eyes_segmentation" / "masks"

# 輸出
METROLOGY_DIR = DATA_DIR / "metrology"
METROLOGY_VIS_DIR = METROLOGY_DIR / "visualizations"
METROLOGY_RESULTS_JSON = METROLOGY_DIR / "metrology_results.json"


# ---------------------------------------------------------------------------
# 距離計算
# ---------------------------------------------------------------------------

def euclidean_distance(p1: dict, p2: dict) -> float:
    """計算兩點 {"x", "y"} 的歐氏距離（像素）。"""
    dx = p1["x"] - p2["x"]
    dy = p1["y"] - p2["y"]
    return math.sqrt(dx * dx + dy * dy)


def get_mask_centroid(mask_path: Path) -> dict | None:
    """
    從 .npz mask 檔案計算幾何中心 (centroid)。

    Returns
    -------
    {"x": float, "y": float} or None (若 mask 為空)
    """
    if not mask_path.exists():
        return None

    mask = np.load(mask_path)["mask"]  # (H, W) bool
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    return {"x": float(xs.mean()), "y": float(ys.mean())}


def get_eye_center(eye_data: dict | None, mask_dir: Path) -> dict | None:
    """
    取得眼睛中心點座標。

    優先級:
    1. keypoint 模型輸出的中心點 (eye_data["center"])
    2. Fallback: mask 的幾何中心 (centroid)

    Returns
    -------
    {"x": float, "y": float, "source": str} or None
    """
    if eye_data is None:
        return None

    # 優先: keypoint center
    center = eye_data.get("center")
    if center and center.get("x") is not None:
        return {"x": center["x"], "y": center["y"], "source": "keypoint"}

    # Fallback: mask centroid
    mask_file = eye_data.get("mask_file")
    if mask_file:
        centroid = get_mask_centroid(mask_dir / mask_file)
        if centroid:
            return {"x": centroid["x"], "y": centroid["y"], "source": "mask_centroid"}

    return None


# ---------------------------------------------------------------------------
# 量測邏輯
# ---------------------------------------------------------------------------

def measure_intra_animal_eye_distance(animal: dict, mask_dir: Path) -> dict:
    """
    量測同一隻動物的雙眼距離。

    Returns
    -------
    dict:
        left_eye_center: {"x", "y", "source"} or None
        right_eye_center: {"x", "y", "source"} or None
        distance_px: float or None
    """
    le_center = get_eye_center(animal.get("left_eye"), mask_dir)
    re_center = get_eye_center(animal.get("right_eye"), mask_dir)

    distance = None
    if le_center and re_center:
        distance = euclidean_distance(le_center, re_center)

    return {
        "left_eye_center": le_center,
        "right_eye_center": re_center,
        "distance_px": distance,
    }


def measure_inter_animal_right_eye_distances(
    animals: list[dict],
    mask_dir: Path,
) -> list[dict]:
    """
    量測同圖中任意兩隻動物的右眼距離（pairwise）。

    Returns
    -------
    list of dict:
        animal_a: {"detection_id", "label"}
        animal_b: {"detection_id", "label"}
        animal_a_right_eye: {"x", "y", "source"} or None
        animal_b_right_eye: {"x", "y", "source"} or None
        distance_px: float or None
    """
    # 收集每隻動物的右眼中心
    animal_right_eyes = []
    for animal in animals:
        re_center = get_eye_center(animal.get("right_eye"), mask_dir)
        animal_right_eyes.append({
            "detection_id": animal["detection_id"],
            "label": animal["label"],
            "right_eye_center": re_center,
        })

    # Pairwise combinations
    pairwise = []
    for a, b in combinations(animal_right_eyes, 2):
        distance = None
        if a["right_eye_center"] and b["right_eye_center"]:
            distance = euclidean_distance(a["right_eye_center"], b["right_eye_center"])

        pairwise.append({
            "animal_a": {"detection_id": a["detection_id"], "label": a["label"]},
            "animal_b": {"detection_id": b["detection_id"], "label": b["label"]},
            "animal_a_right_eye": a["right_eye_center"],
            "animal_b_right_eye": b["right_eye_center"],
            "distance_px": distance,
        })

    return pairwise


# ---------------------------------------------------------------------------
# 視覺化
# ---------------------------------------------------------------------------

def visualize_metrology(
    image_path: Path,
    animals_with_measurements: list[dict],
    pairwise_measurements: list[dict],
    save_path: Path,
) -> None:
    """
    視覺化量測結果：
    - 動物 bbox + 雙眼中心 + 雙眼連線（標註距離）
    - 任意兩隻動物的右眼連線（標註距離）
    """
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)

    fig, ax = plt.subplots(1, figsize=(16, 12))
    ax.imshow(img_array)

    # --- 繪製每隻動物的 bbox + 雙眼 ---
    for idx, animal in enumerate(animals_with_measurements):
        color = [
            (1.0, 0.2, 0.2), (0.2, 0.6, 1.0), (0.2, 1.0, 0.4),
            (1.0, 0.8, 0.1), (0.8, 0.3, 1.0), (1.0, 0.5, 0.1),
        ][idx % 6]
        bbox = animal["bbox"]
        x1, y1, x2, y2 = bbox

        # Bounding box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)

        # Label
        ax.text(
            x1, y1 - 6,
            f"{animal['label']} #{animal['detection_id']}",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(facecolor=color, alpha=0.85, edgecolor="none", pad=2),
        )

        # 雙眼中心點
        intra = animal.get("intra_eye_measurement", {})
        le = intra.get("left_eye_center")
        re = intra.get("right_eye_center")
        dist = intra.get("distance_px")

        if le:
            ax.plot(le["x"], le["y"], "o", color=color, markersize=10,
                    markeredgecolor="white", markeredgewidth=2, zorder=5)
            ax.text(le["x"] - 8, le["y"] - 12, "L", fontsize=8,
                    color="white", fontweight="bold", ha="center",
                    bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=1))

        if re:
            ax.plot(re["x"], re["y"], "s", color=color, markersize=10,
                    markeredgecolor="white", markeredgewidth=2, zorder=5)
            ax.text(re["x"] + 8, re["y"] - 12, "R", fontsize=8,
                    color="white", fontweight="bold", ha="center",
                    bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=1))

        # 雙眼連線 + 距離標註
        if le and re and dist:
            ax.plot([le["x"], re["x"]], [le["y"], re["y"]],
                    "-", color=color, linewidth=2, alpha=0.8, zorder=4)
            mid_x = (le["x"] + re["x"]) / 2
            mid_y = (le["y"] + re["y"]) / 2
            ax.text(mid_x, mid_y + 15,
                    f"{dist:.1f} px",
                    fontsize=9, fontweight="bold", color="white", ha="center",
                    bbox=dict(facecolor=color, alpha=0.85, edgecolor="none", pad=2),
                    zorder=6)

    # --- 繪製 inter-animal 右眼連線 ---
    for pw in pairwise_measurements:
        re_a = pw.get("animal_a_right_eye")
        re_b = pw.get("animal_b_right_eye")
        dist = pw.get("distance_px")

        if re_a and re_b and dist:
            ax.plot(
                [re_a["x"], re_b["x"]], [re_a["y"], re_b["y"]],
                "--", color="cyan", linewidth=2, alpha=0.7, zorder=3,
            )
            mid_x = (re_a["x"] + re_b["x"]) / 2
            mid_y = (re_a["y"] + re_b["y"]) / 2
            label_a = pw["animal_a"]["label"]
            label_b = pw["animal_b"]["label"]
            ax.text(
                mid_x, mid_y - 10,
                f"R↔R: {dist:.1f} px\n({label_a}↔{label_b})",
                fontsize=8, fontweight="bold", color="cyan", ha="center",
                bbox=dict(facecolor="black", alpha=0.7, edgecolor="cyan",
                          linewidth=1, pad=3),
                zorder=6,
            )

    ax.set_axis_off()
    ax.set_title(
        f"{image_path.stem}  —  Eyes Metrology",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 批次處理
# ---------------------------------------------------------------------------

def process_all_images(
    eye_seg_json: Path = EYE_SEG_RESULTS_JSON,
) -> list[dict]:
    """
    批次處理所有圖片：量測雙眼距離。
    """
    # 載入 eye segmentation 結果
    print(f"\n📄 載入 eye segmentation 結果: {eye_seg_json.name}")
    with open(eye_seg_json, "r", encoding="utf-8") as f:
        eye_seg_results = json.load(f)
    print(f"   共 {len(eye_seg_results)} 張圖片")

    all_results = []
    total = len(eye_seg_results)

    for i, entry in enumerate(eye_seg_results):
        image_id = entry["image_id"]
        file_name = entry["file_name"]
        image_path = IMAGES_DIR / file_name

        print(f"\n[{i+1}/{total}] {file_name} (image_id={image_id})")

        animals_with_measurements = []

        # --- 1. Intra-animal: 同一隻動物的雙眼距離 ---
        for animal in entry["animals"]:
            det_id = animal["detection_id"]
            label = animal["label"]

            intra = measure_intra_animal_eye_distance(animal, EYE_MASKS_DIR)
            dist_str = f"{intra['distance_px']:.1f} px" if intra["distance_px"] else "N/A"
            le_src = intra["left_eye_center"]["source"] if intra["left_eye_center"] else "-"
            re_src = intra["right_eye_center"]["source"] if intra["right_eye_center"] else "-"
            print(f"   #{det_id} {label}: 雙眼距離 = {dist_str} (L:{le_src}, R:{re_src})")

            animals_with_measurements.append({
                **animal,
                "intra_eye_measurement": intra,
            })

        # --- 2. Inter-animal: 任意兩隻動物的右眼距離 ---
        pairwise = measure_inter_animal_right_eye_distances(
            entry["animals"], EYE_MASKS_DIR
        )
        for pw in pairwise:
            a = pw["animal_a"]
            b = pw["animal_b"]
            dist_str = f"{pw['distance_px']:.1f} px" if pw["distance_px"] else "N/A"
            print(f"   右眼距離: #{a['detection_id']}({a['label']}) ↔ "
                  f"#{b['detection_id']}({b['label']}) = {dist_str}")

        # --- 視覺化 ---
        if image_path.exists():
            vis_path = METROLOGY_VIS_DIR / f"{image_id}_metrology.jpg"
            visualize_metrology(
                image_path, animals_with_measurements, pairwise, vis_path
            )
            print(f"   視覺化: {vis_path.name}")

        all_results.append({
            "image_id": image_id,
            "file_name": file_name,
            "width": entry["width"],
            "height": entry["height"],
            "animals": animals_with_measurements,
            "inter_animal_right_eye_distances": pairwise,
        })

    # --- 儲存結果 ---
    METROLOGY_DIR.mkdir(parents=True, exist_ok=True)
    with open(METROLOGY_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 結果已儲存: {METROLOGY_RESULTS_JSON}")

    # --- 摘要統計 ---
    total_animals = sum(len(r["animals"]) for r in all_results)
    intra_measured = sum(
        1 for r in all_results
        for a in r["animals"]
        if a["intra_eye_measurement"]["distance_px"] is not None
    )
    inter_measured = sum(
        1 for r in all_results
        for pw in r["inter_animal_right_eye_distances"]
        if pw["distance_px"] is not None
    )
    inter_total = sum(len(r["inter_animal_right_eye_distances"]) for r in all_results)

    print(f"\n{'='*60}")
    print(f" ✅ Eyes Metrology 完成")
    print(f"    處理圖片: {len(all_results)}")
    print(f"    動物數: {total_animals}")
    print(f"    雙眼距離 (intra): {intra_measured}/{total_animals} 成功量測")
    print(f"    右眼距離 (inter): {inter_measured}/{inter_total} 成功量測")
    print(f"    結果: {METROLOGY_RESULTS_JSON}")
    print(f"    視覺化: {METROLOGY_VIS_DIR}")
    print(f"{'='*60}")

    return all_results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = process_all_images()
