# AnyDisc Automated AST Imaging and Reporting System

## Overview

The **AnyDisc Automated AST Imaging and Reporting System** is a Python-based tool that automates the analysis of antibiotic susceptibility test (AST) images. Using a YOLOv7-tiny model, it detects inhibition zones, identifies antibiotic disks, and annotates the results on the image. The processed images and CSV reports provide detailed insights into the detected zones and disk placements.

---

## Features

- **Automated Detection**: Detects inhibition zones and antibiotic disks with high accuracy.
- **Annotation**: Automatically draws annotations on the provided image.
- **CSV Export**: Outputs a detailed CSV file with inhibition zone measurements and antibiotic disk information.
- **Customizable**: The YOLOv7-tiny model allows for further training and expansion for additional antibiotic disk types or zone analyses.

---

## Prerequisites

1. **Python 3.x**: Ensure Python is installed.
2. **Dependencies**: Install required Python packages:
   ```bash
   pip install torch opencv-python-headless
   ```
3. **YOLOv7-tiny Model Weights**: Make sure the model weights (`MIC_best.pt`) and antibiotic data (`Drug_new.json`) are available in the working directory.

---

## Model Details: YOLOv7-Tiny

The model used, **`MIC_best.pt`**, is based on **YOLOv7-tiny**, a lightweight and efficient object detection architecture. It is trained to detect inhibition zones and antibiotic disks in AST images.

- **Future Expansion**:  
   If needed, you can retrain or fine-tune this model on additional datasets to expand its capabilities. This makes it suitable for:
   - Adding more antibiotic types.
   - Improving detection accuracy.
   - Adapting the system to new use cases.
  
- **Transfer Learning**:  
   Use this model as the foundation for further training by leveraging **YOLOv7-tiny**'s architecture and performance. This allows the system to support evolving antibiotic susceptibility testing needs.

To retrain the model, you can gather additional data, annotate it, and fine-tune the model using YOLOv7's standard training procedures.

---

## Usage

### **Command**

```bash
python detect.py \
    --input_image ./data/TestPhoto/2023-08-25_15:18:34.jpg \
    --output_image ./static/output/processed_image_$(date +"%Y-%m-%d_%H-%M-%S").jpg \
    --bac Enterobacterales \
    --dish 9
```

### **Parameters**
- `--input_image`: Path to the input AST image.
- `--output_image`: Path where the processed image will be saved.
- `--bac`: Type of bacteria (default: `Enterobacterales`).
- `--dish`: Diameter of the petri dish in cm (default: 9).

---

## Output

1. **Processed Image**: 
   - Annotated image with inhibition zones and antibiotic disk labels saved at the specified path.

2. **CSV Report**:
   - Detailed CSV output containing information about detected zones and disk placements.

**Sample CSV Output (Inhibition Zones):**
| Radius | Central_x | Central_y | Width | Height | Site | lefttop_x | lefttop_y | rightbottom_x | rightbottom_y |
|--------|-----------|-----------|-------|--------|------|-----------|-----------|--------------|--------------|
| 220    | 344       | 227       | 441   | 447    | O    | 124       | 4         | 565          | 451          |
| 94     | 313       | 301       | 189   | 180    | I    | 218       | 217       | 407          | 397          |

**Sample CSV Output (Antibiotic Disks):**
| Central       | MIC_name | lefttop_x | lefttop_y | rightbottom_x | rightbottom_y |
|---------------|----------|-----------|-----------|---------------|---------------|
| (306, 312)    | ER5      | 287.0     | 292.0     | 326.0         | 332.0         |
| (473, 210)    | CT10     | 455.0     | 191.0     | 491.0         | 229.0         |

---

## Workflow

1. **Image Loading**: Loads the input AST image for processing.
2. **Inhibition Zone Detection**: Detects and measures the inhibition zones.
3. **Antibiotic Disk Recognition**: Identifies the position and name of antibiotic disks.
4. **Annotation**: Draws inhibition zones and disk information onto the image.
5. **CSV Export**: Saves detection results in a CSV file.

---

## Error Handling

- **Missing Model Weights**: If the `MIC_best.pt` file is missing:
  ```
  Error: Model weights 'MIC_best.pt' not found
  ```
- **Invalid Image Path**: If the input image path is incorrect:
  ```
  Error: Unable to read the image at ./data/TestPhoto/invalid_image.jpg
  ```

---

## Directory Structure

```
AnyDisc-Public/
├── detect.py
├── MIC_best.pt
├── Drug_new.json
├── data/
│   └── TestPhoto/
├── static/
│   └── output/
│       ├── processed_image_<timestamp>.jpg
│       ├── result_<timestamp>.csv
│       └── <other_images>.jpg
```

---

## Future Development

The system's use of YOLOv7-tiny as the base model opens opportunities for further development and customization. 

- **Additional Training**:  
   Collect new data and fine-tune the model to improve its detection capabilities for new antibiotics or special cases.
   
- **Integration with Other Systems**:  
   Extend the tool's usage to integrate with lab management systems, automating the reporting process further.

- **Performance Enhancements**:  
   The lightweight nature of YOLOv7-tiny ensures fast processing, but it can also be scaled up by retraining with more complex architectures if needed.

---

## License

This project is open-source. Feel free to use, modify, and distribute it according to the project’s license.

---

## Conclusion

The **AnyDisc Automated AST Imaging and Reporting System** streamlines the AST result interpretation process by leveraging a YOLOv7-tiny model for accurate inhibition zone detection and antibiotic disk recognition. With future training opportunities and easy-to-understand outputs, it serves as a valuable tool for researchers and healthcare professionals.

---

Feel free to reach out if you encounter any issues or have suggestions for improvements.
