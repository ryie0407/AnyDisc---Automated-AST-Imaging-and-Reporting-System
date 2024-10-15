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

def Find_circle_radius(image):
    # Define the save path for output images
    save_path = './static/output'
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # Create the directory if it doesn't exist

    # Copy the original image to avoid modifications to it
    img = image.copy()

    # Ensure that the image is correctly provided
    if img is None:
        print("Error: No image provided")
        return None

    # Convert the image from BGR to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define the HSV range for yellow color
    lower_yellow = np.array([5, 5, 0])
    upper_yellow = np.array([40, 255, 255])
    # Create a mask for yellow color
    mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    # Save the yellow mask image
    cv2.imwrite(os.path.join(save_path, '1.mask_yellow.jpg'), mask_yellow)

    # Find contours in the yellow mask
    contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return None

    # Identify the largest contour as the dish contour
    max_contour = max(contours, key=cv2.contourArea)
    # Create a mask to extract the dish area
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(mask, [max_contour], -1, (255, 255, 255), -1)
    # Retain only the dish area in the image
    masked_image = cv2.bitwise_and(img, mask)

    # Calculate the average HSV value within the dish contour
    mean_val = cv2.mean(hsv_img, mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
    print("Average HSV value in the contour: (Hue, Saturation, Value) =", mean_val[:3])

    # Adjust the mask range based on the average HSV value to detect inhibition zones
    lower_hue = max(10, mean_val[0] - 10)
    upper_hue = min(40, mean_val[0] + 10)
    lower_saturation = max(20, mean_val[1] - 10)
    upper_saturation = min(60, mean_val[1] + 20)
    lower_value = max(150, mean_val[2] - 30)
    upper_value = min(200, mean_val[2] + 50)

    lower_green = np.array([lower_hue, lower_saturation, lower_value])
    upper_green = np.array([upper_hue, upper_saturation, upper_value])

    # Apply Gaussian blur to the dish area and convert to HSV
    blur = cv2.GaussianBlur(masked_image, (15, 15), 0)
    img_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # Create a green mask to detect inhibition zones
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
    # Save the green mask image
    cv2.imwrite(os.path.join(save_path, '2.mask_green.jpg'), mask_green)

    # Define a kernel and apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opening = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)
    # Save the gradient image
    cv2.imwrite(os.path.join(save_path, '3.gradient.jpg'), gradient)

    # Find contours in the gradient image
    contours, _ = cv2.findContours(gradient, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_image = img.copy()  # Prepare the final image for drawing circles

    # Initialize lists to store data
    Central_x, Central_y, Radius, width, height, LT_x, LT_y, RB_x, RB_y, Site = [], [], [], [], [], [], [], [], [], []

    # Calculate the center and radius of the dish
    M = cv2.moments(max_contour)
    center_x = int(M['m10'] / M['m00'])
    center_y = int(M['m01'] / M['m00'])
    plate_radius = max(np.sqrt((center_x - x)**2 + (center_y - y)**2) for x, y in max_contour[:, 0, :])

    # Process each contour to identify inhibition zones
    if contours:
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            distance_from_center = np.sqrt((center_x - x)**2 + (center_y - y)**2)
            # Ensure the circle is fully contained within the dish
            if cv2.contourArea(contour) > 1000 and (distance_from_center + radius) <= plate_radius:
                cv2.circle(final_image, (int(x), int(y)), int(radius), (255, 0, 0), 2)  # Draw circles
                x, y, w, h = cv2.boundingRect(contour)
                radius = int(w / 2)
                if w > 40:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        Central_x.append(cX)
                        Central_y.append(cY)
                        Radius.append(radius)
                        width.append(w)
                        height.append(h)
                        LT_x.append(x)
                        LT_y.append(y)
                        RB_x.append(x + w)
                        RB_y.append(y + h)
                        Site.append("O" if w > 400 else "I")

    # Save the final image with circles
    cv2.imwrite(os.path.join(save_path, '4.circle.jpg'), final_image)

    # Create a DataFrame to store the results and save it to CSV
    CD_list = pd.DataFrame({
        'Radius': Radius,
        'Central_x': Central_x,
        'Central_y': Central_y,
        'width': width,
        'height': height,
        'Site': Site,
        'lefttop_x': LT_x,
        'lefttop_y': LT_y,
        'rightbottom_x': RB_x,
        'rightbottom_y': RB_y,
    })

    CD_list.to_csv(os.path.join(save_path, 'CD.csv'), index=False)
    CD_list.to_csv('CD.csv', index=False)

    return CD_list
def Recognize_location_and_name(image_path, device, model, imgsz=640, stride=32):
    dataset = LoadImages(image_path, img_size=imgsz, stride=int(model.stride.max()))
    names = model.module.names if hasattr(model, 'module') else model.names
    # img為縮放過後的圖像，img0為原始圖像 
    Central, LT_x, LT_y, RB_x, RB_y, MIC_name = [],[],[],[],[],[]
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, 0.40, 0.40, agnostic=False)
        # det = [x1, y1, x2, y2, conf, class]
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 將原本640的位置改成原始位置
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                # Write results
                for x1, y1, x2, y2, conf, cls in reversed(det.numpy()):
                    label = names[int(cls)]
                    cX, cY = int((x1 + x2 )/2), int((y1 + y2 )/2)
                    Central.append((cX, cY))
                    LT_x.append(x1)
                    LT_y.append(y1)
                    RB_x.append(x2)
                    RB_y.append(y2)
                    MIC_name.append(label)
                    
    d_list = pd.DataFrame({
        'Central':Central,
        'MIC_name':MIC_name,
        'lefttop_x':LT_x,
        'lefttop_y':LT_y,
        'rightbottom_x':RB_x,
        'rightbottom_y':RB_y,

    }) 
    d_list.to_csv('./d.csv',index=False)
    return d_list

def ContainedBox(boxA, boxB):
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB
    
    if x1B >= x1A and y1B >= y1A and x2B <= x2A and y2B <= y2A:
        return True
    else:
        return False
    
def DrawMIC(image, image_path, CD_list, d_list, bac, MIC, dish):
    img6 = image.copy()
    if len(d_list) == 0:
        return False
    Mname, Locate, Bioname, Acronym, Biocon, Biomm, Bioresist, imagename, dishes = [], [], [], [], [], [], [], [], []
    for i in range(len(CD_list)):
        Radius, Central_x, Central_y, width, height, Site, *xyxy = CD_list.iloc[i]
        Mname.append('-')
        Locate.append('I')
        if Site !='O':
            for j in range(len(d_list)):
                MIC_Central, MIC_name, *MICxyxy = d_list.iloc[j]
                InBox = ContainedBox(xyxy, MICxyxy)
                if InBox:
                    Mname[i] = MIC_name
                    Locate[i] = 'MIC'

        else:
            Locate[i] = 'O'
            dish_pix = width

    CD_list['MIC_name'] = Mname
    CD_list['Site'] = Locate
    
    Disk_name = CD_list['MIC_name'].unique()[np.where(CD_list['MIC_name'].unique() != '-')]
    print(len(Disk_name))
    Bacs = []
    for i in Disk_name:
        if len(Disk_name) < 1:
            return False
            
            break
        imagename.append(image_path)
        Draw_list = CD_list.loc[CD_list['MIC_name']==i]
        Bacs.append(bac)
#         mm = np.round((Draw_list[Draw_list['Site']=='MIC']['width'].mean()/dish_pix)*dish*10,1).values[0]
        mm = np.round((Draw_list[Draw_list['Site']=='MIC']['width'].mean()/dish_pix)*dish*10,1)
        dishes.append(dish)
        Biomm.append(mm)
        
        Radius = int(Draw_list[Draw_list['Site']=='MIC']['Radius'].mean())
        x, y = (int(Draw_list[Draw_list['Site']=='MIC']['Central_x'].mean()), int(Draw_list[Draw_list['Site']=='MIC']['Central_y'].mean()))
        shift = 30
        img_width = img6.shape[1]
        
        Re = MIC[i][bac]    
        if Re == '-':
            Resistance = 'ND'
            color = (0, 0, 0)
        elif mm <= Re[0]:
            Resistance='R'
            Color=(20, 20, 200)
        elif mm >= Re[1]:
            Resistance='S'
            Color=(20, 200, 20)
        else:
            Resistance='I'
            Color=(20, 200, 20)
        Bioresist.append(Resistance)
        
        if i == 'ER5':
            Bioname.append('Enrofloxacin (ER 5)')
            Acronym.append('ER')
            Biocon.append('5')
        elif i == 'CT10':
            Bioname.append('Colistin (CT 10)')
            Acronym.append('CT')
            Biocon.append('10')
        else:
            Bioname.append('Ceftiofur (CF 30)')
            Acronym.append('CF')
            Biocon.append('30')

        cv2.line(img6, (x - Radius, y ), (x + Radius, y), (156, 188, 24), 2)
        cv2.putText(img6, "{} mm".format(mm), (x - 60, y - (shift + ((img_width//40)*2 + 10))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img6, "{}".format(i), (x - 60, y - (shift + ((img_width//40)*3 + 10))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img6, "{}".format(Resistance), (x - 15, y - shift), cv2.FONT_HERSHEY_SIMPLEX, 1, Color, 2, cv2.LINE_AA)
    
    cv2.imwrite(image_path, img6)

    with open('Flow.txt', 'a')as f:
        f.write('b')
    with open('Flow.txt', 'r')as f:
        count = len(f.read())
        b = '0'
        c = '00'
        if count < 10:
            count = c + str(count)
        elif count >= 10 and count < 100:
            count = b + str(count)
        else:
            count = str(count)
    counts = []
    for i in range(len(Disk_name)):
        counts.append(count)
    
    csv_list = pd.DataFrame({
        'A':Bioname,
        'B':Acronym,
        'C':Biocon,
        'D':Biomm,
        'E':Bioresist,
        'F':imagename,
        'G':dishes,
        'H':Bacs,
        'I':counts
        
    })
    csv_list.to_csv('./MIC.csv')
    return count