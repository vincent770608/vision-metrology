"""
data_filter.py — COCO Dataset Animal Image Filter

從 COCO 資料集中篩選包含 2 隻（含）以上動物的圖片。
使用 COCO val2017 annotations。
"""

import json
import os
import zipfile
from pathlib import Path
from collections import Counter

import requests

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
# 專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 資料暫存目錄
DATA_DIR = PROJECT_ROOT / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
IMAGES_DIR = DATA_DIR / "images" / "val2017"

# COCO 2017 Val 資源
ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
ANNOTATIONS_ZIP = DATA_DIR / "annotations_trainval2017.zip"
INSTANCES_JSON = ANNOTATIONS_DIR / "instances_val2017.json"

# COCO val2017 圖片下載 base URL
IMAGE_BASE_URL = "http://images.cocodataset.org/val2017"

# COCO 動物 supercategory 下的所有類別
ANIMAL_SUPERCATEGORY = "animal"

# 最少動物數量門檻
MIN_ANIMAL_COUNT = 2

# # 最小標註面積 (像素²) — 太小的動物看不到眼睛，在此步驟就過濾掉
MIN_ANNOTATION_AREA = 3000

# 輸出結果目錄
OUTPUT_DIR = DATA_DIR / "filtered"

# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """下載檔案，帶進度顯示。"""
    if dest.exists():
        print(f"  ✓ 已存在: {dest.name}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ 下載中: {url}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r    {pct:5.1f}%  ({downloaded:,} / {total:,} bytes)", end="")
    print()


def extract_annotations(zip_path: Path, extract_to: Path) -> None:
    """解壓標註檔案（只解壓 instances_val2017.json）。"""
    target_name = "annotations/instances_val2017.json"
    out_file = extract_to / "instances_val2017.json"

    if out_file.exists():
        print(f"  ✓ 已解壓: {out_file.name}")
        return

    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ 解壓中: {target_name}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # 只解壓需要的那一個
        for member in zf.namelist():
            if member == target_name:
                data = zf.read(member)
                with open(out_file, "wb") as f:
                    f.write(data)
                print(f"  ✓ 完成: {out_file}")
                return

    raise FileNotFoundError(f"{target_name} 未在 zip 中找到")


def download_image(image_info: dict, dest_dir: Path) -> Path:
    """下載單張 COCO 圖片。"""
    file_name = image_info["file_name"]
    dest = dest_dir / file_name
    if dest.exists():
        return dest

    url = f"{IMAGE_BASE_URL}/{file_name}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    return dest


# ---------------------------------------------------------------------------
# 核心篩選邏輯
# ---------------------------------------------------------------------------

def filter_images_with_multiple_animals(
    json_path: Path,
    min_count: int = MIN_ANIMAL_COUNT,
) -> list[dict]:
    """
    篩選出包含 >= min_count 隻動物的圖片。

    回傳格式:
    [
        {
            "image_id": int,
            "file_name": str,
            "width": int,
            "height": int,
            "animal_count": int,
            "animals": [{"annotation_id", "category_id", "category_name", "bbox", "area"}, ...],
        },
        ...
    ]
    """

    """載入 COCO JSON 標註。"""
    print(f"\n📂 載入標註: {json_path.name}")
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    print(f"   images:      {len(coco['images']):,}")
    print(f"   annotations: {len(coco['annotations']):,}")
    print(f"   categories:  {len(coco['categories'])}")

    """取得所有 supercategory == 'animal' 的類別 id → name 對應。"""
    animal_cats = {
        cat["id"]: cat["name"]
        for cat in coco["categories"]
        if cat["supercategory"] == ANIMAL_SUPERCATEGORY
    }
    print(f"\n🐾 動物類別 ({len(animal_cats)} 個):")
    for cid, name in sorted(animal_cats.items()):
        print(f"   [{cid:>2}] {name}")

    # 建立 image_id → image info 對照表
    image_lookup = {img["id"]: img for img in coco["images"]}

    # 收集每張圖片中對應的動物 annotation
    image_animals: dict[int, list[dict]] = {}
    skipped_small = 0
    skipped_crowd = 0
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in animal_cats:
            continue
        # 跳過 crowd 標註（不代表單隻動物）
        if ann.get("iscrowd", 0):
            skipped_crowd += 1
            continue
        # 跳過面積太小的動物（太遠/太小，看不到眼睛）
        if ann["area"] < MIN_ANNOTATION_AREA:
            skipped_small += 1
            continue
        img_id = ann["image_id"]
        if img_id not in image_animals:
            image_animals[img_id] = []
        image_animals[img_id].append(
            {
                "annotation_id": ann["id"],
                "category_id": cat_id,
                "category_name": animal_cats[cat_id],
                "bbox": ann["bbox"],  # [x, y, w, h]
                "area": ann["area"],
            }
        )
    # print(f"\n   已過濾: {skipped_small} 隻太小, {skipped_crowd} 個 crowd 標註")

    # 篩選
    results = []
    for img_id, animals in image_animals.items():
        if len(animals) >= min_count:
            img_info = image_lookup[img_id]
            results.append(
                {
                    "image_id": img_id,
                    "file_name": img_info["file_name"],
                    "width": img_info["width"],
                    "height": img_info["height"],
                    "coco_url": img_info.get("coco_url", ""),
                    "animal_count": len(animals),
                    "animals": animals,
                }
            )

    # 按動物數量降序排列
    results.sort(key=lambda x: x["animal_count"], reverse=True)

    print(f"\n📊 篩選結果:")
    print(f"   共 {len(results):,} 張圖片包含 ≥ {min_count} 隻動物")

    """將篩選結果存為 JSON。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "filtered_animal_images.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 結果已儲存: {out_file}")

    # 統計
    counter = Counter()
    for r in results:
        for a in r["animals"]:
            counter[a["category_name"]] += 1
    print(f"\n   各類動物出現次數:")
    for name, cnt in counter.most_common():
        print(f"     {name:<12} {cnt:>5}")

    return results


def download_images(results: list[dict], n: int = 5) -> list[Path]:
    """下載前 n 張篩選結果圖片，方便目視確認。"""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for entry in results:
        img_info = {"file_name": entry["file_name"]}
        path = download_image(img_info, IMAGES_DIR)
        animal_names = [a["category_name"] for a in entry["animals"]]
        print(f"{entry['file_name']}  — {entry['animal_count']} 隻動物: {animal_names}")
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # print("=" * 60)
    # print(" COCO Vision Metrology — Step 1: 篩選含多隻動物的圖片")
    # print("=" * 60)
    #
    # # 1. 下載標註
    # print("\n[1/4] 下載 COCO 2017 Val 標註 ...")
    # download_file(ANNOTATIONS_URL, ANNOTATIONS_ZIP)
    #
    # # 2. 解壓
    # print("\n[2/4] 解壓標註檔案 ...")
    # extract_annotations(ANNOTATIONS_ZIP, ANNOTATIONS_DIR)

    # 3. 篩選 & 儲存結果
    print("\n[3/4] 篩選圖片 ...")
    results = filter_images_with_multiple_animals(INSTANCES_JSON)
    #
    # 4. 下載樣本
    print("\n[4/4] 儲存結果 & 下載樣本圖片 ...")
    sample_paths = download_images(results, n=5)

    print("\n" + "=" * 60)
    print(f" ✅ 完成！共篩選出 {len(results)} 張圖片")
    print(f"    結果 JSON: {OUTPUT_DIR / 'filtered_animal_images.json'}")
    print(f"    樣本圖片:  {IMAGES_DIR}")
    print("=" * 60)
