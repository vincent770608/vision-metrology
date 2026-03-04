"""
export_hrnet_onnx.py — 將 HRNet-w32 PyTorch 模型轉為 ONNX 格式

使用方式:
    python scripts/export_hrnet_onnx.py
"""

import sys
from pathlib import Path

# 加入專案根目錄到 path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from app.services.keypoint_detection import HRNet, remap_checkpoint_keys
from app.core.config import (
    HRNET_CKPT_FILE, HRNET_ONNX_FILE, HRNET_INPUT_SIZE, NUM_KEYPOINTS,
)


def export_to_onnx():
    print("=" * 60)
    print(" HRNet-w32 AP-10K → ONNX Export")
    print("=" * 60)

    # 1. 建立模型
    print("\n[1/3] 建立 HRNet-w32 模型 ...")
    model = HRNet(num_keypoints=NUM_KEYPOINTS)

    # 2. 載入 checkpoint
    print(f"[2/3] 載入權重: {HRNET_CKPT_FILE.name}")
    if not HRNET_CKPT_FILE.exists():
        print(f"  ⚠ 權重檔不存在: {HRNET_CKPT_FILE}")
        print("  請先執行 keypoint_detection.py 下載權重")
        return

    checkpoint = torch.load(HRNET_CKPT_FILE, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = remap_checkpoint_keys(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"  模型參數量: {params:,}")

    # 3. Export ONNX
    print(f"[3/3] 匯出 ONNX: {HRNET_ONNX_FILE.name}")
    H, W = HRNET_INPUT_SIZE
    dummy_input = torch.randn(1, 3, H, W)

    HRNET_ONNX_FILE.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(HRNET_ONNX_FILE),
        input_names=["image"],
        output_names=["heatmaps"],
        dynamic_axes={
            "image": {0: "batch"},
            "heatmaps": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    size_mb = HRNET_ONNX_FILE.stat().st_size / 1024 / 1024
    print(f"\n  ✅ 匯出完成: {HRNET_ONNX_FILE}")
    print(f"     檔案大小: {size_mb:.1f} MB")

    # 驗證
    import onnxruntime as ort
    import numpy as np

    print("\n  驗證 ONNX Runtime 推論 ...")
    session = ort.InferenceSession(str(HRNET_ONNX_FILE))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run(
        [output_name],
        {input_name: np.random.randn(1, 3, H, W).astype(np.float32)},
    )
    print(f"  Output shape: {result[0].shape}")
    print(f"  ✅ ONNX Runtime 驗證通過")


if __name__ == "__main__":
    export_to_onnx()
