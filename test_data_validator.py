#!/usr/bin/env python3
"""
test_data_validator.py — 驗證 test_data.csv 中的測試數據格式和計算

使用方式:
    python test_data_validator.py
"""

import csv
import math
from pathlib import Path


def euclidean_distance(x1, y1, x2, y2) -> float:
    """計算歐氏距離。"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def validate_test_data():
    """驗證 test_data.csv 中的所有計算。"""
    csv_file = Path(__file__).parent / "test_data.csv"

    if not csv_file.exists():
        print("❌ test_data.csv 不存在")
        return False

    print("=" * 80)
    print(" 🧪 測試數據驗證")
    print("=" * 80)

    all_valid = True
    errors = []

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader, start=2):  # 跳過標題行
            image_id = row["image_id"]
            image_filename = row["image_filename"]
            num_animals = int(row["num_animals"])
            notes = row.get("notes", "")

            print(f"\n📷 Row {row_idx}: {image_filename}")
            print(f"   動物數: {num_animals} ({row['animal_label_1']}, {row['animal_label_2']})")
            print(f"   說明: {notes}")

            # 驗證動物 1 的距離
            try:
                left_x_1 = float(row["left_eye_x_1"])
                left_y_1 = float(row["left_eye_y_1"])
                right_x_1 = float(row["right_eye_x_1"])
                right_y_1 = float(row["right_eye_y_1"])
                expected_1 = float(row["expected_inter_eye_distance_1"])

                actual_1 = euclidean_distance(left_x_1, left_y_1, right_x_1, right_y_1)
                error_1 = abs(actual_1 - expected_1)

                if error_1 > 0.5:  # 允許 0.5 像素的容差
                    print(f"  ❌ 動物 1 ({row['animal_label_1']}): 預期 {expected_1:.2f} px, 實際 {actual_1:.2f} px")
                    errors.append(f"Row {row_idx} Animal 1: {error_1:.2f} px error")
                    all_valid = False
                else:
                    print(f"  ✅ 動物 1 ({row['animal_label_1']}): {actual_1:.2f} px (預期 {expected_1:.2f} px)")

            except (KeyError, ValueError) as e:
                print(f"  ❌ 動物 1: 數據缺失或錯誤 ({e})")
                all_valid = False

            # 驗證動物 2 的距離
            try:
                left_x_2 = float(row["left_eye_x_2"])
                left_y_2 = float(row["left_eye_y_2"])
                right_x_2 = float(row["right_eye_x_2"])
                right_y_2 = float(row["right_eye_y_2"])
                expected_2 = float(row["expected_inter_eye_distance_2"])

                actual_2 = euclidean_distance(left_x_2, left_y_2, right_x_2, right_y_2)
                error_2 = abs(actual_2 - expected_2)

                if error_2 > 0.5:
                    print(f"  ❌ 動物 2 ({row['animal_label_2']}): 預期 {expected_2:.2f} px, 實際 {actual_2:.2f} px")
                    errors.append(f"Row {row_idx} Animal 2: {error_2:.2f} px error")
                    all_valid = False
                else:
                    print(f"  ✅ 動物 2 ({row['animal_label_2']}): {actual_2:.2f} px (預期 {expected_2:.2f} px)")

            except (KeyError, ValueError) as e:
                print(f"  ❌ 動物 2: 數據缺失或錯誤 ({e})")
                all_valid = False

            # 驗證 pairwise 距離
            try:
                right_eye_x_1 = float(row["right_eye_x_1"])
                right_eye_y_1 = float(row["right_eye_y_1"])
                right_eye_x_2 = float(row["right_eye_x_2"])
                right_eye_y_2 = float(row["right_eye_y_2"])
                expected_pairwise = float(row["expected_right_eye_distance_pairwise"])

                actual_pairwise = euclidean_distance(
                    right_eye_x_1, right_eye_y_1,
                    right_eye_x_2, right_eye_y_2
                )

                error_pair = abs(actual_pairwise - expected_pairwise)
                if error_pair > 0.5:
                    print(f"  ❌ Pairwise 右眼: 預期 {expected_pairwise:.2f} px, 實際 {actual_pairwise:.2f} px")
                    errors.append(f"Row {row_idx} Pairwise: {error_pair:.2f} px error")
                    all_valid = False
                else:
                    print(f"  ✅ Pairwise 右眼: {actual_pairwise:.2f} px (預期 {expected_pairwise:.2f} px)")

            except (KeyError, ValueError) as e:
                print(f"  ❌ Pairwise 距離: 數據缺失或錯誤 ({e})")
                all_valid = False


    print("\n" + "=" * 80)
    if all_valid:
        print(" ✅ 所有測試數據驗證通過！")
        print(f"   共 {row_idx - 1} 組測試數據，0 個錯誤")
    else:
        print(" ❌ 存在驗證失敗的行")
        print(f"   錯誤列表:")
        for error in errors:
            print(f"     - {error}")
    print("=" * 80)

    return all_valid


if __name__ == "__main__":
    success = validate_test_data()
    exit(0 if success else 1)
