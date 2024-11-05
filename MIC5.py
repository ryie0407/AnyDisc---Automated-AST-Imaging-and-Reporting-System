import argparse
import os
import operator
import cv2
import time
import numpy as np
import pandas as pd
import torch

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

## 1030
def Find_circle_radius(image, save_path, image_name):
    # Ensure the output directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create a copy of the original image to avoid modifying it directly
    img = image.copy()
    if img is None:
        print("Error: No image provided")
        return None

    # Remove file extension, keep only the image name
    image_base_name = os.path.splitext(image_name)[0]

    # Convert the image to HSV color space and create a yellow mask
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([5, 5, 0])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    cv2.imwrite(os.path.join(save_path, f'{image_base_name}_1.mask_yellow.jpg'), mask_yellow)

    # Find contours in the yellow mask and select the largest contour for the Petri dish
    contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return None

    max_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(mask, [max_contour], -1, (255, 255, 255), -1)
    masked_image = cv2.bitwise_and(img, mask)

    # Calculate the average HSV value of the yellow region
    mean_val = cv2.mean(hsv_img, mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
    mean_hue, mean_saturation, mean_value = mean_val[0], mean_val[1], mean_val[2]

    # Dynamically adjust the range for the green mask based on the average HSV values
    hue_offset = 8 + (4.5 if mean_hue >= 27 else 2.5)
    saturation_offset = 15 + (3.9 if mean_saturation >= 29.8 else 2.0)
    value_offset = 25 + (5.5 if mean_value >= 199 else 3.0)

    # Set the lower and upper limits for the green mask, adding flexibility
    lower_green = np.array([
        max(14, mean_hue - hue_offset),
        max(20, mean_saturation - saturation_offset),
        max(140, mean_value - value_offset)
    ])
    upper_green = np.array([
        min(37, mean_hue + hue_offset),
        min(54, mean_saturation + saturation_offset),
        min(210, mean_value + value_offset)
    ])

    # Apply the dynamically set green mask range
    blur = cv2.GaussianBlur(masked_image, (15, 15), 0)
    img_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
    cv2.imwrite(os.path.join(save_path, f'{image_base_name}_2.mask_green.jpg'), mask_green)

    # Process the mask using morphological closing to fill in fragmented edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_mask = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    # Use morphological gradient to highlight edges, removing small noise again
    gradient = cv2.morphologyEx(closed_mask, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite(os.path.join(save_path, f'{image_base_name}_3.gradient.jpg'), gradient)

    # Find contours and filter out contours with small areas
    contours, _ = cv2.findContours(gradient, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_image = img.copy()
    Central_x, Central_y, Radius, width, height, LT_x, LT_y, RB_x, RB_y, Site = [], [], [], [], [], [], [], [], [], []

    # Calculate the center and radius of the Petri dish
    M = cv2.moments(max_contour)
    if M["m00"] != 0:
        center_x = int(M['m10'] / M['m00'])
        center_y = int(M['m01'] / M['m00'])
    else:
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2  # Default to image center
    plate_radius = max(np.sqrt((center_x - x)**2 + (center_y - y)**2) for x, y in max_contour[:, 0, :])

    # Add the Petri dish information to the list
    x, y, w, h = cv2.boundingRect(max_contour)
    Central_x.append(center_x)
    Central_y.append(center_y)
    Radius.append(int(plate_radius))
    width.append(w)
    height.append(h)
    LT_x.append(x)
    LT_y.append(y)
    RB_x.append(x + w)
    RB_y.append(y + h)
    Site.append('O')  # Mark as Petri dish

    # No longer drawing Petri dish contours to avoid showing them in the output image

    # Process the remaining contours (inhibition zones)
    inhibition_zones = []
    for contour in contours:
        # Skip the Petri dish contour
        if np.array_equal(contour, max_contour):
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        distance_from_center = np.sqrt((center_x - x)**2 + (center_y - y)**2)
        contour_area = cv2.contourArea(contour)
        # Add condition to exclude contours with diameters greater than 70mm (convert 70mm to pixels)
        # Since actual pixel-to-mm ratio is missing, assume maximum allowed radius is some pixel value, e.g., 350 pixels
        if contour_area > 1000 and (distance_from_center + radius) <= plate_radius and radius * 2 <= 350:
            inhibition_zones.append({
                'contour': contour,
                'center': (int(x), int(y)),
                'radius': int(radius),
                'area': contour_area
            })

    # Merge overlapping or close inhibition zones, keeping only the largest one
    filtered_inhibition_zones = []
    for zone in inhibition_zones:
        center = zone['center']
        radius = zone['radius']
        duplicate = False
        for existing_zone in filtered_inhibition_zones:
            existing_center = existing_zone['center']
            distance = np.sqrt((center[0] - existing_center[0])**2 + (center[1] - existing_center[1])**2)
            if distance < 20:  # Threshold can be adjusted as needed
                # Keep the inhibition zone with a larger radius
                if radius > existing_zone['radius']:
                    existing_zone.update(zone)
                duplicate = True
                break
        if not duplicate:
            filtered_inhibition_zones.append(zone)

    # Process filtered inhibition zones
    for idx, zone in enumerate(filtered_inhibition_zones):
        contour = zone['contour']
        x, y = zone['center']
        radius = zone['radius']
        # Draw inhibition zone on the image
        cv2.circle(final_image, (x, y), radius, (255, 0, 0), 1)  # Blue circle indicates inhibition zone

        # Draw diameter line
        cv2.line(final_image, (x - radius, y), (x + radius, y), (0, 255, 255), 4)  # Yellow line indicates diameter
        # Add diameter text
        diameter = radius * 2
        cv2.putText(final_image, f"Diameter: {diameter}px", (x - radius, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Collect data
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            Central_x.append(cX)
            Central_y.append(cY)
            Radius.append(radius)
            width.append(w_rect)
            height.append(h_rect)
            LT_x.append(x_rect)
            LT_y.append(y_rect)
            RB_x.append(x_rect + w_rect)
            RB_y.append(y_rect + h_rect)
            Site.append('I')  # Mark as inhibition zone

    cv2.imwrite(os.path.join(save_path, f'{image_base_name}_4.circle.jpg'), final_image)

    # Save detection results to CSV
    CD_list = pd.DataFrame({
        'Radius': Radius, 'Central_x': Central_x, 'Central_y': Central_y, 'width': width,
        'height': height, 'Site': Site, 'lefttop_x': LT_x, 'lefttop_y': LT_y,
        'rightbottom_x': RB_x, 'rightbottom_y': RB_y,
    })
    CD_list.to_csv(os.path.join(save_path, f'{image_base_name}_CD.csv'), index=False)

    return CD_list

def Recognize_location_and_name(image_path, device, model, save_path, imgsz=640, stride=32):
    # Get image name
    image_base_name = os.path.splitext(os.path.basename(image_path))[0]
    # Load image dataset
    dataset = LoadImages(image_path, img_size=imgsz, stride=int(model.stride.max()))
    # Get model's class names
    names = model.module.names if hasattr(model, 'module') else model.names
    Central, LT_x, LT_y, RB_x, RB_y, MIC_name, confidences = [], [], [], [], [], [], []

    for path, img, im0s, vid_cap in dataset:
        print("Model input image shape:", img.shape)
        print("Original image shape:", im0s.shape)

        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = non_max_suppression(model(img, augment=False)[0], 0.4, 0.4, agnostic=False)
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                print("Scaled detection coordinates:")
                for x1, y1, x2, y2, conf, cls in det.cpu().numpy():
                    print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
                    label = names[int(cls)]
                    cX, cY = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    Central.append((cX, cY))
                    LT_x.append(int(x1))
                    LT_y.append(int(y1))
                    RB_x.append(int(x2))
                    RB_y.append(int(y2))
                    MIC_name.append(label)
                    confidences.append(conf)

    # Save detection results to DataFrame
    d_list = pd.DataFrame({
        'Central': Central, 'MIC_name': MIC_name, 'confidence': confidences,
        'lefttop_x': LT_x, 'lefttop_y': LT_y,
        'rightbottom_x': RB_x, 'rightbottom_y': RB_y
    })

    # Check if there are two CF30 and missing CT10, if so, change the higher confidence CF30 to CT10
    cf30_rows = d_list[d_list["MIC_name"] == "CF30"]
    if len(cf30_rows) > 1 and "CT10" not in d_list["MIC_name"].values and "ER5" in d_list["MIC_name"].values:
        highest_cf30_idx = cf30_rows["confidence"].idxmax()
        d_list.at[highest_cf30_idx, "MIC_name"] = "CT10"
        print("Marked the CF30 with higher confidence as CT10 to correct the missing CT10 situation")

    # Save final results to CSV
    d_list.to_csv(os.path.join(save_path, f'{image_base_name}_d.csv'), index=False)
    return d_list

def is_center_close(centerA, centerB, threshold):
    xA, yA = centerA
    xB, yB = centerB
    distance = np.sqrt((xA - xB) ** 2 + (yA - yB) ** 2)
    return distance <= threshold

# Overlapping centers
def DrawMIC(image, image_path, save_path, CD_list, d_list, bac, MIC, dish):
    img6 = image.copy()
    if len(d_list) == 0:
        print("d_list is empty.")
        return False

    # Get Petri dish information
    dish_row = CD_list[CD_list['Site'] == 'O']
    if dish_row.empty:
        print("No dish detected.")
        return False
    elif len(dish_row) > 1:
        print("Multiple dishes detected. Please check the detection algorithm.")
        return False
    else:
        dish_pix = dish_row['width'].values[0]
        print(f"Dish width (dish_pix): {dish_pix}")

    # Remove Petri dish information for subsequent processing
    CD_list_no_dish = CD_list[CD_list['Site'] != 'O']

    # Initialize Mname and Locate lists
    Mname = []
    Locate = []
    threshold_distance = 20  # Threshold distance for center matching

    # Add a step to overlap the green center with the red point (antibiotic disc center)
    for idx, row in CD_list_no_dish.iterrows():
        # Get inhibition zone information
        inhibition_center = (row['Central_x'], row['Central_y'])
        inhibition_radius = row['Radius']
        
        matched = False  # Used to mark if a matching antibiotic disc is found
        # Match antibiotic discs
        for jdx, d_row in d_list.iterrows():
            MIC_Central, MIC_name = d_row['Central'], d_row['MIC_name']
            mic_center = MIC_Central  # Center of antibiotic disc

            # Determine if the inhibition zone and disc are in the same area and align
            distance = np.sqrt((inhibition_center[0] - mic_center[0]) ** 2 + (inhibition_center[1] - mic_center[1]) ** 2)
            if distance <= inhibition_radius:
                print(f"Align inhibition zone {idx} with antibiotic {MIC_name} concentric")

                # Calculate movement
                delta_x = mic_center[0] - inhibition_center[0]
                delta_y = mic_center[1] - inhibition_center[1]

                # Update the center position of the green point
                CD_list_no_dish.loc[idx, 'Central_x'] = mic_center[0]
                CD_list_no_dish.loc[idx, 'Central_y'] = mic_center[1]

                # Move the diameter line position
                LT_x = row['lefttop_x'] + delta_x
                LT_y = row['lefttop_y'] + delta_y
                RB_x = row['rightbottom_x'] + delta_x
                RB_y = row['rightbottom_y'] + delta_y
                CD_list_no_dish.loc[idx, 'lefttop_x'] = LT_x
                CD_list_no_dish.loc[idx, 'lefttop_y'] = LT_y
                CD_list_no_dish.loc[idx, 'rightbottom_x'] = RB_x
                CD_list_no_dish.loc[idx, 'rightbottom_y'] = RB_y

                # Update Mname and Locate lists
                Mname.append(MIC_name)
                Locate.append('MIC')
                matched = True
                break  # If aligned, do not look for other antibiotic discs

        # If no matching antibiotic disc is found, use default values
        if not matched:
            Mname.append('-')
            Locate.append('I')

    # Update configuration information to CD_list
    CD_list_no_dish['MIC_name'] = Mname
    CD_list_no_dish['Site'] = Locate

    # Draw inhibition zones and disc-related information on the image
    for idx, row in CD_list_no_dish.iterrows():
        x = int(row['Central_x'])
        y = int(row['Central_y'])
        Radius = int(row['Radius'])
        MIC_name = row['MIC_name']
        
        diameter_mm = round((Radius * 2 / dish_pix) * dish * 10, 1)  # Calculate diameter in mm

        # Exclude inhibition zones with a diameter greater than 70mm
        if diameter_mm > 80:
            continue
        
        # Draw inhibition zone with diameter line
        # cv2.circle(img6, (x, y), Radius, (255, 0, 0), 2)  # Blue contour
        cv2.line(img6, (x - Radius, y), (x + Radius, y), (156, 188, 24), 2)  # Morandi green diameter line
        cv2.putText(img6, f"{diameter_mm} mm", (x - 60, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (156, 188, 24), 2)
        
        # Check if there is a corresponding antibiotic and bacteria combination in the MIC library
        if MIC_name in MIC and bac in MIC[MIC_name]:
            Re = MIC[MIC_name][bac]
        else:
            Re = '-'

        # Determine resistance based on inhibition zone diameter
        if Re == '-':
            Resistance = 'ND'
            Color = (0, 0, 0)
        elif diameter_mm <= Re[0]:
            Resistance = 'R'
            Color = (20, 20, 200)
        elif diameter_mm >= Re[1]:
            Resistance = 'S'
            Color = (20, 200, 20)
        else:
            Resistance = 'I'
            Color = (20, 200, 20)

        # Display only antibiotic abbreviation and resistance determination on the image
        cv2.putText(img6, f"{MIC_name}", (x - 60, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img6, f"{Resistance}", (x - 15, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, Color, 2, cv2.LINE_AA)

    # Draw red and green dots to mark the centers of antibiotic discs and inhibition zones
    for idx, row in CD_list_no_dish.iterrows():
        x = int(row['Central_x'])
        y = int(row['Central_y'])
        cv2.circle(img6, (x, y), 5, (0, 255, 0), -1)  # Center of inhibition zone, green

    for idx, row in d_list.iterrows():
        xMIC, yMIC = int(row['Central'][0]), int(row['Central'][1])
        cv2.circle(img6, (xMIC, yMIC), 5, (0, 0, 255), -1)  # Center of antibiotic disc, red

    # Save the final image to the specified directory
    output_image_path = os.path.join(save_path, os.path.basename(image_path))
    cv2.imwrite(output_image_path, img6)

    return True
