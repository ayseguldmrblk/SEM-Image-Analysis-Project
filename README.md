# Porosity Detection App

The Porosity Detection App is a simple application built using PyQt5 and PyTorch with Ultralytics YOLO for porosity detection in SEM images. The app allows users to select an image, performs porosity detection, displays the result with bounding boxes, and saves the annotated image.

## Project Files

- **analyzer.py**: Main Python script containing the Porosity Detection App.
  
- **porosity_model.pt**: YOLO model file for porosity detection. (Added using Git LFS)
  
- **sem_images/**: Directory containing SEM images for testing.
  
- **results/**: Directory to store annotated images. (Generated by the app)
  
- **requirements.txt**: List of project dependencies. Use the following command to install them:

    ```bash
    pip install -r requirements.txt
    ```
  
- **test.py**: Python script for testing. 

## Prerequisites

Before running the Porosity Detection App, ensure you have the required dependencies installed. You can install them using the following:

```bash
pip install -r requirements.txt
```

Make sure you have the YOLO model file (porosity_model.pt) in the same directory as the application script.

## How to Run

To run the Porosity Detection App, execute the following command in your terminal:

```bash
python analyzer.py
```
## Usage

- Launch the application.
- Click the "Select Image" button to choose an SEM image.
- The app will perform porosity detection and display the image with bounding boxes.
- The aspect ratio and classification of the detected porosity will be shown.
- The annotated image will be saved in the "results" directory with the original file name appended with "_analysis".

## Notes
- This application assumes a YOLO model for porosity detection (porosity_model.pt). Make sure to have the model file in the same directory.
- Adjust the positioning of the added text in the script if needed.
