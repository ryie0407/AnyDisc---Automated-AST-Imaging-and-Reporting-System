# detect.py

import os
import argparse
import torch
import json
import cv2
from MIC5 import Find_circle_radius, Recognize_location_and_name, DrawMIC

def process_image(input_image, output_image, bac, dish, MIC, output_folder):
    # Read the image
    pic = cv2.imread(input_image)
    if pic is None:
        print(f"Error: Unable to read image {input_image}")
        return False

    # Check if the model weights file exists
    weights = 'MIC_best.pt'
    if not os.path.isfile(weights):
        print(f"Error: Model weights file '{weights}' does not exist")
        return False

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    try:
        model = torch.load(weights, map_location=device)['model'].float().eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error: Unable to load model '{weights}', error message: {e}")
        return False

    # Extract original image name
    image_name = os.path.basename(input_image)

    # 1. Detect inhibition zones in the image
    CD_list = Find_circle_radius(pic, output_folder, image_name)
    if CD_list is None or CD_list.empty:
        print("Error: No inhibition zones detected")
        return False

    # 2. Identify the location and name of antibiotic disks
    d_list = Recognize_location_and_name(input_image, device, model, output_folder)
    if d_list is None or d_list.empty:
        print("Error: Unable to identify disk locations and names")
        return False

    # 3. Draw inhibition zones and annotations on the image
    success = DrawMIC(pic, output_image, output_folder, CD_list, d_list, bac, MIC, dish)
    if not success:
        print("Error: Failed to draw annotations on the image")
        return False

    print(f"Processed image saved to: {output_image}")
    return True

def process_folder(input_folder, output_folder, bac, dish, MIC):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_image = os.path.join(input_folder, filename)
            output_image = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_output.jpg")

            print(f"Processing image: {filename}")
            success = process_image(input_image, output_image, bac, dish, MIC, output_folder)
            if success:
                print(f"{filename} processed successfully")
            else:
                print(f"{filename} processing failed")

def main():
    parser = argparse.ArgumentParser(description='Batch process images in a folder and generate annotation results')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input image folder')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to save processed images and CSVs')
    parser.add_argument('--bac', type=str, default="Enterobacterales", help='Bacteria type')
    parser.add_argument('--dish', type=int, default=9, help='Diameter of the culture dish (cm)')
    args = parser.parse_args()

    MIC = json.load(open('Drug_new.json', 'r', encoding='utf-8'))
    process_folder(args.input_folder, args.output_folder, args.bac, args.dish, MIC)

if __name__ == "__main__":
    main()
