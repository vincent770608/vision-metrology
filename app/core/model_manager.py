"""
model_manager.py — 模型管理器

在 FastAPI 啟動時：
1. 檢查所有 checkpoint 是否存在，不存在就下載
2. 載入所有模型到記憶體

三個模型:
- Mask R-CNN (torchvision): instance segmentation
- HRNet-w32 (ONNX Runtime): keypoint detection
- SAM vit_b (segment-anything): eye segmentation
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
import torch
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from segment_anything import sam_model_registry, SamPredictor

from app.core.config import (
    MODELS_DIR,
    # Mask R-CNN
    MASKRCNN_SCORE_THRESHOLD, MASK_THRESHOLD,
    ANIMAL_LABEL_IDS, COCO_INSTANCE_CATEGORY_NAMES,
    # HRNet
    HRNET_CKPT_URL, HRNET_CKPT_FILE, HRNET_ONNX_FILE,
    HRNET_INPUT_SIZE, HRNET_HEATMAP_SIZE, NUM_KEYPOINTS,
    LEFT_EYE_IDX, RIGHT_EYE_IDX, KPT_SCORE_THRESHOLD,
    PIXEL_MEAN, PIXEL_STD,
    # SAM
    SAM_TYPE, SAM_CKPT_URL, SAM_CKPT_FILE, EYE_BOX_PADDING,
    # 通用
    MIN_BBOX_AREA,
)


class ModelManager:
    """管理所有模型的生命週期。"""

    def __init__(self):
        self.maskrcnn = None
        self.hrnet_session: ort.InferenceSession | None = None
        self.sam_predictor: SamPredictor | None = None
        self._device = "cpu"

    # ------------------------------------------------------------------
    # Checkpoint 下載
    # ------------------------------------------------------------------

    @staticmethod
    def _download(url: str, dest: Path) -> None:
        if dest.exists():
            print(f"  ✓ {dest.name} 已存在")
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  ↓ 下載 {dest.name} ...")
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r    {downloaded/total*100:5.1f}%", end="")
        print()

    def ensure_checkpoints(self) -> None:
        """啟動時檢查/下載所有 checkpoint。"""
        print("\n📦 檢查模型 checkpoints ...")
        self._download(HRNET_CKPT_URL, HRNET_CKPT_FILE)
        self._download(SAM_CKPT_URL, SAM_CKPT_FILE)
        # Mask R-CNN: torchvision 自動下載，不需要手動處理

        # 檢查 ONNX 是否存在，不存在就匯出
        if not HRNET_ONNX_FILE.exists():
            print("  ⚠ HRNet ONNX 不存在，執行匯出 ...")
            self._export_hrnet_onnx()

    def _export_hrnet_onnx(self) -> None:
        """內部執行 HRNet → ONNX 匯出。"""
        from app.services.keypoint_detection import HRNet, remap_checkpoint_keys

        model = HRNet(num_keypoints=NUM_KEYPOINTS)
        ckpt = torch.load(HRNET_CKPT_FILE, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        sd = remap_checkpoint_keys(sd)
        model.load_state_dict(sd, strict=False)
        model.eval()

        H, W = HRNET_INPUT_SIZE
        dummy = torch.randn(1, 3, H, W)
        HRNET_ONNX_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model, dummy, str(HRNET_ONNX_FILE),
            input_names=["image"], output_names=["heatmaps"],
            dynamic_axes={"image": {0: "batch"}, "heatmaps": {0: "batch"}},
            opset_version=17, do_constant_folding=True,
        )
        print(f"  ✓ ONNX 匯出完成: {HRNET_ONNX_FILE.name}")

    # ------------------------------------------------------------------
    # 模型載入
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """載入所有模型到記憶體。"""
        print("\n🔧 載入所有模型 ...")
        self._load_maskrcnn()
        self._load_hrnet_onnx()
        self._load_sam()
        print("\n✅ 所有模型載入完成\n")

    def _load_maskrcnn(self) -> None:
        print("  [1/3] Mask R-CNN ...")
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.maskrcnn = maskrcnn_resnet50_fpn_v2(weights=weights)
        self.maskrcnn.eval()
        print("        ✓ Mask R-CNN 就緒")

    def _load_hrnet_onnx(self) -> None:
        print("  [2/3] HRNet-w32 ONNX ...")
        self.hrnet_session = ort.InferenceSession(
            str(HRNET_ONNX_FILE),
            providers=["CPUExecutionProvider"],
        )
        print("        ✓ HRNet ONNX 就緒")

    def _load_sam(self) -> None:
        print("  [3/3] SAM vit_b ...")
        sam = sam_model_registry[SAM_TYPE](checkpoint=str(SAM_CKPT_FILE))
        sam.to(device=self._device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)
        print("        ✓ SAM 就緒")

    # ------------------------------------------------------------------
    # 推論 API
    # ------------------------------------------------------------------

    def detect_animals(self, image: Image.Image) -> list[dict]:
        """
        用 Mask R-CNN 偵測動物。

        Returns: list of {label, label_id, score, bbox [x1,y1,x2,y2], mask}
        """
        from torchvision.transforms import functional as F

        img_tensor = F.to_tensor(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.maskrcnn(img_tensor)[0]

        animals = []
        for idx in range(len(outputs["labels"])):
            label_id = int(outputs["labels"][idx])
            score = float(outputs["scores"][idx])

            if label_id not in ANIMAL_LABEL_IDS:
                continue
            if score < MASKRCNN_SCORE_THRESHOLD:
                continue

            bbox = outputs["boxes"][idx].tolist()  # [x1, y1, x2, y2]
            mask = outputs["masks"][idx, 0].numpy() > MASK_THRESHOLD  # (H, W) bool

            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < MIN_BBOX_AREA:
                continue

            animals.append({
                "label": COCO_INSTANCE_CATEGORY_NAMES[label_id],
                "label_id": label_id,
                "score": score,
                "bbox": bbox,
                "mask": mask,
            })
        return animals

    def detect_eyes_onnx(
        self, image: Image.Image, bbox: list[float],
    ) -> dict:
        """
        用 HRNet ONNX 偵測眼睛 keypoints。

        Returns: {left_eye, right_eye} 各自為 {x, y, confidence} or None
        """
        from torchvision import transforms

        x1, y1, x2, y2 = bbox
        w, h = image.size
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        crop = image.crop((x1, y1, x2, y2))

        transform = transforms.Compose([
            transforms.Resize(HRNET_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
        ])
        tensor = transform(crop).unsqueeze(0).numpy()  # (1, 3, 256, 256)

        input_name = self.hrnet_session.get_inputs()[0].name
        output_name = self.hrnet_session.get_outputs()[0].name
        heatmaps = self.hrnet_session.run(
            [output_name], {input_name: tensor.astype(np.float32)}
        )[0][0]  # (K, 64, 64)

        bbox_w, bbox_h = x2 - x1, y2 - y1
        K, hm_h, hm_w = heatmaps.shape

        def extract_kpt(k_idx):
            hm = heatmaps[k_idx]
            flat = np.argmax(hm)
            hy, hx = divmod(flat, hm_w)
            conf = float(hm[hy, hx])
            if conf < KPT_SCORE_THRESHOLD:
                return None
            orig_x = x1 + (hx / hm_w) * bbox_w
            orig_y = y1 + (hy / hm_h) * bbox_h
            return {"x": float(orig_x), "y": float(orig_y), "confidence": conf}

        return {
            "left_eye": extract_kpt(LEFT_EYE_IDX),
            "right_eye": extract_kpt(RIGHT_EYE_IDX),
        }

    def segment_eye(
        self, eye_point: dict, bbox: list[float],
        img_w: int, img_h: int,
    ) -> dict | None:
        """
        用 SAM 分割單隻眼睛。

        Returns: {mask_area_px, sam_score, eye_box} or None
        """
        if self.sam_predictor is None:
            return None

        cx, cy = eye_point["x"], eye_point["y"]
        pad = EYE_BOX_PADDING
        ax1, ay1, ax2, ay2 = bbox

        ex1 = max(cx - pad, ax1, 0)
        ey1 = max(cy - pad, ay1, 0)
        ex2 = min(cx + pad, ax2, img_w)
        ey2 = min(cy + pad, ay2, img_h)

        point_coords = np.array([[cx, cy]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        eye_box = np.array([ex1, ey1, ex2, ey2], dtype=np.float32)

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=eye_box,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        return {
            "mask_area_px": int(masks[best_idx].sum()),
            "sam_score": float(scores[best_idx]),
            "eye_box": [float(ex1), float(ey1), float(ex2), float(ey2)],
        }


# 全域 singleton
model_manager = ModelManager()
