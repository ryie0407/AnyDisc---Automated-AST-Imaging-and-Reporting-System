# AnyDisc Automated AST Imaging and Reporting System

## Overview

The **AnyDisc Automated AST Imaging and Reporting System** is a Python-based tool for automated analysis of antibiotic susceptibility test (AST) images. It uses a YOLO-based model to detect inhibition zones, recognize antibiotic disks, and annotate results directly onto the provided images. The system outputs processed images and corresponding result files in CSV format.

---

## Features

- **Automated Detection**: Identifies inhibition zones and antibiotic disks using a pre-trained YOLO model.
- **Annotation**: Annotates the processed image with zone and disk information.
- **CSV Export**: Outputs detailed information, including inhibition zone radius and disk position, in CSV format.
- **Device Agnostic**: Runs on either GPU or CPU based on system availability.

---

## Prerequisites

1. **Python 3.x**: Ensure Python is installed on your system.
2. **Dependencies**: Install the required Python packages using:
   ```bash
   pip install torch opencv-python-headless
   ```
3. **Model Weights**: Ensure the YOLO model weights (`MIC_best.pt`) and antibiotic range data (`Drug_new.json`) are present in the working directory.

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
- `--input_image`: Path to the input AST image (e.g., `./data/TestPhoto/2023-08-25_15:18:34.jpg`).
- `--output_image`: Path where the processed image will be saved (e.g., `./static/output/processed_image_2024-10-15_17-38-11.jpg`).
- `--bac`: Type of bacteria (default: `Enterobacterales`).
- `--dish`: Diameter of the petri dish in cm (default: 9).

---

## Output

1. **Processed Image**: 
   - Annotated image showing inhibition zones and antibiotic disks saved at the specified path (e.g., `./static/output/processed_image_2024-10-15_17-38-11.jpg`).

2. **CSV Report**:
   - CSV file containing detailed information about detected inhibition zones and antibiotic disks.
   - Example output saved as `./static/output/result_20241015_173812.csv`.

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
4. **Annotation**: Draws the inhibition zones and labels onto the image.
5. **CSV Export**: Saves detailed detection results in CSV format.

---

## Example Output

- **Processed Image**:  
  `./static/output/processed_image_2024-10-15_17-38-11.jpg`

- **CSV Result File**:  
  `./static/output/result_20241015_173812.csv`

---

## Error Handling

- **Missing Model Weights**: If the `MIC_best.pt` file is missing, the program will display:
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

## License

This project is open-source. Feel free to use, modify, and distribute it as per the project's license.

---

## Conclusion

The **AnyDisc Automated AST Imaging and Reporting System** simplifies the process of AST result interpretation by automating inhibition zone detection and antibiotic disk recognition. With easy-to-use commands and clear outputs, it is a powerful tool for researchers and healthcare professionals alike.

---

Feel free to reach out if you encounter any issues or have suggestions for improvements.
