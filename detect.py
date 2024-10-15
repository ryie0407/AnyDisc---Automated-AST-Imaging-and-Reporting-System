# detect.py

import os
import argparse
import datetime
import torch
import json
import cv2
from MIC5 import Find_circle_radius, Recognize_location_and_name, DrawMIC

def process_image(input_image, output_image, bac, dish, MIC):
    # 讀取圖像
    pic = cv2.imread(input_image)
    if pic is None:
        print(f"錯誤：無法讀取圖像 {input_image}")
        return False

    # 檢查模型權重檔案是否存在
    weights = 'MIC_best.pt'
    if not os.path.isfile(weights):
        print(f"錯誤：模型權重檔案 '{weights}' 不存在")
        return False

    # 設定設備 (GPU 或 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入 YOLO 模型
    try:
        model = torch.load(weights, map_location=device)['model'].float().eval()
        print("模型成功載入")
    except Exception as e:
        print(f"錯誤：無法載入模型 '{weights}'，錯誤訊息：{e}")
        return False

    # 創建圖像副本進行處理
    image_for_detection = pic.copy()
    image_for_drawing = pic.copy()

    # 1. 檢測圖像中的抑菌圈
    CD_list = Find_circle_radius(image_for_detection)
    if CD_list is None or CD_list.empty:
        print("錯誤：未檢測到抑菌圈")
        return False

    print("成功檢測抑菌圈:", CD_list)

    # 2. 識別抗生素紙錠的位置與名稱
    d_list = Recognize_location_and_name(input_image, device, model)
    if d_list is None or d_list.empty:
        print("錯誤：未能識別紙錠位置與名稱")
        return False

    print("成功識別紙錠資訊:", d_list)

    # 3. 繪製抑菌圈與標註在圖像上
    success = DrawMIC(image_for_drawing, output_image, CD_list, d_list, bac, MIC, dish)
    if not success:
        print("錯誤：在圖像上繪製標註失敗")
        return False

    print(f"處理後的圖像已儲存至: {output_image}")

    # 4. 儲存 CSV 檔案（包含處理結果）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_csv = f"./static/output/result_{timestamp}.csv"
    CD_list.to_csv(result_csv, index=False)
    print(f"結果資料已儲存至: {result_csv}")

    return True

def main():
    # 解析命令列參數
    parser = argparse.ArgumentParser(description='處理圖像並產生標註結果')
    parser.add_argument('--input_image', type=str, required=True, help='輸入圖像的路徑')
    parser.add_argument('--output_image', type=str, required=True, help='處理後圖像的儲存路徑')
    parser.add_argument('--bac', type=str, default="Enterobacterales", help='細菌類型')
    parser.add_argument('--dish', type=int, default=9, help='培養皿的直徑（cm）')
    args = parser.parse_args()

    # 載入 MIC 抗生素範圍資料
    MIC = json.load(open('Drug_new.json', 'r', encoding='utf-8'))

    # 執行圖像處理
    success = process_image(args.input_image, args.output_image, args.bac, args.dish, MIC)
    if success:
        print("圖像處理成功完成")
    else:
        print("圖像處理失敗")

if __name__ == "__main__":
    main()
