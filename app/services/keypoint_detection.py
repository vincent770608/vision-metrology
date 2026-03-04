"""
keypoint_detection.py — 動物眼睛 Keypoint Detection (Pure PyTorch)

使用 HRNet-w32 (AP-10K pretrained) 在每隻動物的 ROI (bbox) 內
偵測 left_eye 和 right_eye 的像素座標。

不依賴 MMPose 框架，直接用 PyTorch 載入模型權重進行推論。

AP-10K 17 Keypoints (0-indexed):
  0: L_Eye  ← 使用
  1: R_Eye  ← 使用
  2: Nose
  3-16: 其他身體關鍵點
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images" / "val2017"

# 輸入：instance segmentation 結果
SEG_RESULTS_JSON = DATA_DIR / "segmentation" / "segmentation_results.json"

# 輸出
KPT_DIR = DATA_DIR / "keypoints"
KPT_VIS_DIR = KPT_DIR / "visualizations"
KPT_RESULTS_JSON = KPT_DIR / "keypoint_results.json"

# 模型
MODEL_DIR = PROJECT_ROOT / "app" / "models"
HRNET_CKPT_URL = "https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_ap10k_256x256-18aac840_20211029.pth"
HRNET_CKPT_FILE = MODEL_DIR / "hrnet_w32_ap10k_256x256-18aac840_20211029.pth"

# 輸入圖片大小 (AP-10K 訓練時的設定)
INPUT_SIZE = (256, 256)  # (H, W)
HEATMAP_SIZE = (64, 64)  # (H, W) = INPUT_SIZE / 4

# AP-10K keypoint indices
NUM_KEYPOINTS = 17
LEFT_EYE_IDX = 0
RIGHT_EYE_IDX = 1
KPT_SCORE_THRESHOLD = 0.3

# 最小 bbox 面積 (像素²) — 太小的動物看不到眼睛，跳過
MIN_BBOX_AREA = 3000

# 前處理用的 mean/std (ImageNet)
PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]

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


# ============================================================================
# HRNet-w32 Architecture (Standalone PyTorch Implementation)
# ============================================================================

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super().__init__()
        self.num_branches = num_branches
        self.fuse_method = fuse_method
        self.multi_scale_output = multi_scale_output
        self.num_inchannels = num_inchannels

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i],
                                  kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                            ))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True),
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + torchF.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HRNet(nn.Module):
    """
    HRNet-w32 for top-down heatmap-based pose estimation.
    Configuration matches MMPose's td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.
    """

    def __init__(self, num_keypoints=17):
        super().__init__()

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1 (layer1): 4 Bottleneck blocks, 64->256
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        # Transition from stage1 to stage2
        self.transition1 = self._make_transition_layer([256], [32, 64])

        # Stage 2: 1 module, 2 branches
        self.stage2, pre_stage_channels = self._make_stage(
            num_modules=1, num_branches=2, num_blocks=[4, 4],
            num_channels=[32, 64], block=BasicBlock,
            num_inchannels=[32, 64])

        # Transition from stage2 to stage3
        self.transition2 = self._make_transition_layer(pre_stage_channels, [32, 64, 128])

        # Stage 3: 4 modules, 3 branches
        self.stage3, pre_stage_channels = self._make_stage(
            num_modules=4, num_branches=3, num_blocks=[4, 4, 4],
            num_channels=[32, 64, 128], block=BasicBlock,
            num_inchannels=[32, 64, 128])

        # Transition from stage3 to stage4
        self.transition3 = self._make_transition_layer(pre_stage_channels, [32, 64, 128, 256])

        # Stage 4: 3 modules, 4 branches
        # multi_scale_output=False: 最後一個 module 只輸出最高解析度 branch
        # (與 MMPose top-down heatmap config 一致)
        self.stage4, pre_stage_channels = self._make_stage(
            num_modules=3, num_branches=4, num_blocks=[4, 4, 4, 4],
            num_channels=[32, 64, 128, 256], block=BasicBlock,
            num_inchannels=[32, 64, 128, 256],
            multi_scale_output=False)

        # Final prediction head: 1x1 conv from high-res branch (32ch) to keypoints
        self.final_layer = nn.Conv2d(32, num_keypoints, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []

        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i],
                                  3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True),
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, num_modules, num_branches, num_blocks, num_channels,
                    block, num_inchannels, multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(HighResolutionModule(
                num_branches, block, num_blocks, num_inchannels,
                num_channels, 'SUM', reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Stage 1
        x = self.layer1(x)

        # Transition 1
        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        # Stage 2
        y_list = self.stage2(x_list)

        # Transition 2
        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        # Stage 3
        y_list = self.stage3(x_list)

        # Transition 3
        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        # Stage 4
        y_list = self.stage4(x_list)

        # Head: only use highest resolution branch
        out = self.final_layer(y_list[0])
        return out


# ============================================================================
# Checkpoint Key Mapping
# ============================================================================

def remap_checkpoint_keys(state_dict: dict) -> dict:
    """
    將 MMPose checkpoint 的 key 名稱對應到我們的 HRNet 架構。
    MMPose 的 key 格式為 'backbone.xxx' 和 'head.xxx'(或 'keypoint_head.xxx')。
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key

        # 移除 'backbone.' 前綴
        if new_key.startswith("backbone."):
            new_key = new_key[len("backbone."):]

        # 將 head 層對應到 final_layer
        if key.startswith("keypoint_head.") or key.startswith("head."):
            # MMPose head: keypoint_head.final_layer.weight / .bias
            #           or head.deconv_layers... / head.final_layer...
            if "final_layer" in key:
                suffix = key.split("final_layer")[-1]  # .weight or .bias
                new_key = f"final_layer{suffix}"
            else:
                continue  # skip deconv layers we don't have

        new_state_dict[new_key] = value

    return new_state_dict


# ============================================================================
# 模型載入
# ============================================================================

def download_checkpoint(url: str, dest: Path) -> None:
    """下載模型權重檔。"""
    if dest.exists():
        print(f"  ✓ 權重已存在: {dest.name}")
        return

    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ 下載權重: {dest.name}")
    resp = requests.get(url, stream=True, timeout=120)
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


def load_pose_model(device: str = "cpu") -> tuple:
    """
    載入 HRNet-w32 AP-10K 模型 (純 PyTorch)。

    Returns
    -------
    (model, device)
    """
    print(f"\n🔧 載入 HRNet-w32 (AP-10K) — Pure PyTorch ...")
    print(f"   Device: {device}")

    # 下載 checkpoint
    download_checkpoint(HRNET_CKPT_URL, HRNET_CKPT_FILE)

    # 建立模型
    model = HRNet(num_keypoints=NUM_KEYPOINTS)

    # 載入權重
    print(f"  📦 載入權重: {HRNET_CKPT_FILE.name}")
    checkpoint = torch.load(HRNET_CKPT_FILE, map_location=device, weights_only=False)

    # MMPose checkpoint 可能是 {'state_dict': ...} 或直接是 state_dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remap keys
    state_dict = remap_checkpoint_keys(state_dict)

    # 載入，strict=False 以容許遺失的 key
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  ⚠ Missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"  ⚠ Unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    model.to(device)
    model.eval()
    print("   ✓ 模型載入完成")

    return model, torch.device(device)


# ============================================================================
# 前處理 & 推論
# ============================================================================

# 前處理 transform
_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),  # (256, 256)
    transforms.ToTensor(),
    transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
])


def crop_and_preprocess(image: Image.Image, bbox: list[float]) -> torch.Tensor:
    """
    從原圖中裁切 bbox 區域，resize 到 INPUT_SIZE，並正規化。

    Parameters
    ----------
    image : PIL Image (RGB)
    bbox : [x1, y1, x2, y2]

    Returns
    -------
    torch.Tensor (1, 3, H, W)
    """
    x1, y1, x2, y2 = bbox
    # 確保邊界不超出圖片
    w, h = image.size
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    # 裁切
    crop = image.crop((x1, y1, x2, y2))

    # 前處理
    tensor = _transform(crop)
    return tensor.unsqueeze(0)  # (1, 3, 256, 256)


def heatmap_to_keypoints(heatmaps: np.ndarray, bbox: list[float]) -> list[dict]:
    """
    從 heatmap 提取 keypoint 座標並轉回原圖座標。

    Parameters
    ----------
    heatmaps : (K, H_hm, W_hm) numpy array
    bbox : [x1, y1, x2, y2] 原圖上的 bbox

    Returns
    -------
    list of dict: [{"x", "y", "confidence"}, ...] 共 K 個
    """
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    K, hm_h, hm_w = heatmaps.shape
    keypoints = []

    for k in range(K):
        hm = heatmaps[k]
        # 找最大值位置
        flat_idx = np.argmax(hm)
        hm_y, hm_x = divmod(flat_idx, hm_w)
        confidence = float(hm[hm_y, hm_x])

        # 轉回原圖座標
        orig_x = x1 + (hm_x / hm_w) * bbox_w
        orig_y = y1 + (hm_y / hm_h) * bbox_h

        keypoints.append({
            "x": float(orig_x),
            "y": float(orig_y),
            "confidence": confidence,
        })

    return keypoints


def detect_keypoints_in_roi(
    model: nn.Module,
    image: Image.Image,
    bbox: list[float],
    device: torch.device,
) -> dict:
    """
    在單一動物 bbox ROI 中偵測 keypoints。

    Returns
    -------
    dict:
        left_eye: {"x", "y", "confidence"} or None
        right_eye: {"x", "y", "confidence"} or None
        all_keypoints: list of 17 keypoints
    """
    # 裁切 & 前處理
    input_tensor = crop_and_preprocess(image, bbox).to(device)

    # 推論
    with torch.no_grad():
        heatmaps = model(input_tensor)  # (1, K, 64, 64)

    heatmaps = heatmaps.cpu().numpy()[0]  # (K, 64, 64)

    # 提取 keypoints
    all_kpts = heatmap_to_keypoints(heatmaps, bbox)

    # 提取 left_eye 和 right_eye
    def extract_eye(idx):
        kpt = all_kpts[idx]
        if kpt["confidence"] >= KPT_SCORE_THRESHOLD:
            return kpt
        return None

    return {
        "left_eye": extract_eye(LEFT_EYE_IDX),
        "right_eye": extract_eye(RIGHT_EYE_IDX),
        "all_keypoints": all_kpts,
    }


# ============================================================================
# 視覺化
# ============================================================================

def visualize_keypoints(
    image_path: Path,
    animals: list[dict],
    save_path: Path,
) -> None:
    """
    視覺化：在原圖上畫出 bbox + 眼睛標記 + 連線。
    """
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)

    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(img_array)

    for idx, animal in enumerate(animals):
        color = VIS_COLORS[idx % len(VIS_COLORS)]
        bbox = animal["bbox"]
        x1, y1, x2, y2 = bbox

        # Bounding box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)

        # Label
        label_text = f"{animal['label']} #{idx}"
        ax.text(
            x1, y1 - 6, label_text,
            fontsize=11, fontweight="bold", color="white",
            bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=2),
        )

        # 眼睛標記
        le = animal.get("left_eye")
        re = animal.get("right_eye")

        if le:
            ax.plot(le["x"], le["y"], "o", color=color, markersize=10,
                    markeredgecolor="white", markeredgewidth=2)
            ax.text(le["x"] + 5, le["y"] - 8, f"L {le['confidence']:.2f}",
                    fontsize=9, color=color, fontweight="bold")

        if re:
            ax.plot(re["x"], re["y"], "s", color=color, markersize=10,
                    markeredgecolor="white", markeredgewidth=2)
            ax.text(re["x"] + 5, re["y"] - 8, f"R {re['confidence']:.2f}",
                    fontsize=9, color=color, fontweight="bold")

        # 雙眼連線
        if le and re:
            ax.plot(
                [le["x"], re["x"]], [le["y"], re["y"]],
                "-", color=color, linewidth=2, alpha=0.7,
            )

    ax.set_axis_off()
    ax.set_title(
        f"{image_path.stem}  —  {len(animals)} animal(s), eye keypoints",
        fontsize=13,
    )
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# 批次處理
# ============================================================================

def process_all_images(
    seg_results_json: Path = SEG_RESULTS_JSON,
    device: str = "cpu",
) -> list[dict]:
    """
    批次處理所有圖片：對每隻動物的 ROI 執行 keypoint detection。
    """
    # 載入 segmentation 結果
    print(f"\n📄 載入 segmentation 結果: {seg_results_json.name}")
    with open(seg_results_json, "r", encoding="utf-8") as f:
        seg_results = json.load(f)
    print(f"   共 {len(seg_results)} 張圖片")

    # 載入 pose model
    model, dev = load_pose_model(device=device)

    all_results = []
    total = len(seg_results)

    for i, entry in enumerate(seg_results):
        image_id = entry["image_id"]
        file_name = entry["file_name"]
        image_path = IMAGES_DIR / file_name

        print(f"\n[{i+1}/{total}] {file_name} (image_id={image_id})")

        if not image_path.exists():
            print(f"   ⚠ 圖片不存在，跳過")
            continue

        # 載入圖片一次，重複使用
        image = Image.open(image_path).convert("RGB")

        animals = []
        for det in entry["detections"]:
            det_id = det["detection_id"]
            label = det["label"]
            bbox = det["bbox"]
            score = det["score"]

            print(f"   動物 #{det_id} ({label}, score={score:.2f}): ", end="")

            # 檢查 bbox 面積 — 太小的動物跳過
            x1, y1, x2, y2 = bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area < MIN_BBOX_AREA:
                print(f"⚠ bbox 太小 ({bbox_area:.0f} px²)，跳過")
                continue

            # 偵測 keypoints
            kpt_result = detect_keypoints_in_roi(model, image, bbox, dev)

            le = kpt_result["left_eye"]
            re = kpt_result["right_eye"]

            le_str = f"({le['x']:.1f}, {le['y']:.1f}) conf={le['confidence']:.2f}" if le else "N/A"
            re_str = f"({re['x']:.1f}, {re['y']:.1f}) conf={re['confidence']:.2f}" if re else "N/A"
            print(f"L_Eye={le_str}  R_Eye={re_str}")

            animal_entry = {
                "detection_id": det_id,
                "label": label,
                "label_id": det.get("label_id"),
                "score": score,
                "bbox": bbox,
                "mask_file": det.get("mask_file"),
                "left_eye": le,
                "right_eye": re,
                "all_keypoints": kpt_result["all_keypoints"],
            }
            animals.append(animal_entry)

        # 視覺化
        vis_path = KPT_VIS_DIR / f"{image_id}_kpt.jpg"
        visualize_keypoints(image_path, animals, vis_path)
        print(f"   視覺化: {vis_path.name}")

        result_entry = {
            "image_id": image_id,
            "file_name": file_name,
            "width": entry["width"],
            "height": entry["height"],
            "animals": animals,
        }
        all_results.append(result_entry)

    # 儲存結果
    KPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(KPT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 結果已儲存: {KPT_RESULTS_JSON}")

    # 摘要統計
    total_animals = sum(len(r["animals"]) for r in all_results)
    eyes_found = sum(
        1 for r in all_results
        for a in r["animals"]
        if a["left_eye"] and a["right_eye"]
    )
    print(f"\n{'='*60}")
    print(f" ✅ Keypoint Detection 完成")
    print(f"    處理圖片: {len(all_results)}")
    print(f"    偵測動物: {total_animals}")
    print(f"    雙眼皆偵測到: {eyes_found}")
    print(f"    結果: {KPT_RESULTS_JSON}")
    print(f"    視覺化: {KPT_VIS_DIR}")
    print(f"{'='*60}")

    return all_results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = process_all_images(device="cpu")
