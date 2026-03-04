"""
metrology.py — POST /v1/metrology/animal-eyes

上傳一張圖片，回傳圖中所有動物的座標與距離量測值。
"""

import io
import math
from itertools import combinations

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from app.core.model_manager import model_manager

router = APIRouter()


def euclidean_distance(p1: dict, p2: dict) -> float:
    dx = p1["x"] - p2["x"]
    dy = p1["y"] - p2["y"]
    return math.sqrt(dx * dx + dy * dy)


@router.post("/animal-eyes")
async def animal_eyes_metrology(file: UploadFile = File(...)):
    """
    上傳一張圖片，回傳動物眼睛偵測與距離量測。

    Pipeline:
    1. Mask R-CNN → 偵測動物 bbox + mask
    2. HRNet ONNX → 偵測每隻動物的雙眼 keypoint
    3. SAM → 眼睛輪廓分割
    4. 量測雙眼距離 + pairwise 右眼距離
    """
    # 驗證檔案類型
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(400, f"不支援的檔案格式: {file.content_type}")

    # 讀取圖片
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "無法解析圖片檔案")

    img_w, img_h = image.size

    # --- Step 1: Mask R-CNN — 偵測動物 ---
    detections = model_manager.detect_animals(image)

    if not detections:
        return {
            "image_info": {"width": img_w, "height": img_h},
            "animals": [],
            "pairwise_right_eye_distances": [],
            "message": "未偵測到動物",
        }

    # 設定 SAM image embedding (一次)
    img_array = np.array(image)
    model_manager.sam_predictor.set_image(img_array)

    # --- Step 2 & 3: 對每隻動物進行眼睛偵測 + 分割 ---
    animals = []
    for det_idx, det in enumerate(detections):
        bbox = det["bbox"]

        # HRNet ONNX → 眼睛 keypoints
        eyes = model_manager.detect_eyes_onnx(image, bbox)

        # SAM → 眼睛輪廓分割
        left_eye_seg = None
        right_eye_seg = None
        if eyes["left_eye"]:
            left_eye_seg = model_manager.segment_eye(
                eyes["left_eye"], bbox, img_w, img_h
            )
        if eyes["right_eye"]:
            right_eye_seg = model_manager.segment_eye(
                eyes["right_eye"], bbox, img_w, img_h
            )

        # 雙眼距離
        inter_eye_distance = None
        if eyes["left_eye"] and eyes["right_eye"]:
            inter_eye_distance = round(
                euclidean_distance(eyes["left_eye"], eyes["right_eye"]), 2
            )

        animal_entry = {
            "id": det_idx,
            "label": det["label"],
            "score": round(det["score"], 4),
            "bbox": [round(v, 1) for v in bbox],
            "left_eye": _format_eye(eyes["left_eye"], left_eye_seg),
            "right_eye": _format_eye(eyes["right_eye"], right_eye_seg),
            "inter_eye_distance_px": inter_eye_distance,
        }
        animals.append(animal_entry)

    # --- Step 4: Pairwise 右眼距離 ---
    pairwise = []
    for a, b in combinations(animals, 2):
        re_a = a.get("right_eye")
        re_b = b.get("right_eye")
        dist = None
        if re_a and re_b:
            dist = round(euclidean_distance(
                {"x": re_a["x"], "y": re_a["y"]},
                {"x": re_b["x"], "y": re_b["y"]},
            ), 2)
        pairwise.append({
            "animal_a_id": a["id"],
            "animal_b_id": b["id"],
            "animal_a_label": a["label"],
            "animal_b_label": b["label"],
            "distance_px": dist,
        })

    return {
        "image_info": {"width": img_w, "height": img_h},
        "animals": animals,
        "pairwise_right_eye_distances": pairwise,
    }


def _format_eye(eye_kpt: dict | None, eye_seg: dict | None) -> dict | None:
    """格式化單隻眼睛的輸出。"""
    if eye_kpt is None:
        return None

    result = {
        "x": round(eye_kpt["x"], 2),
        "y": round(eye_kpt["y"], 2),
        "confidence": round(eye_kpt["confidence"], 4),
    }
    if eye_seg:
        result["mask_area_px"] = eye_seg["mask_area_px"]
        result["sam_score"] = round(eye_seg["sam_score"], 4)
    return result
