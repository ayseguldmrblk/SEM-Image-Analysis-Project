import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt
import torch
from ultralytics import YOLO

class PorosityApp(QWidget):
    def __init__(self):
        super().__init__()

        # Load the PyTorch model
        self.model = YOLO('porosity_model.pt')

        # Initialize aspect ratios list and weights
        self.aspect_ratios = []
        self.weights = []  # Add this line to initialize the weights

        self.init_ui()

    def init_ui(self):
        # Create layout
        layout = QVBoxLayout()

        # Create QLabel to display image
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        # Create QLabel for displaying aspect ratio classification
        self.aspect_ratio_label = QLabel(self)
        layout.addWidget(self.aspect_ratio_label)

        # Create QPushButton for selecting an image
        select_button = QPushButton('Select Image', self)
        select_button.clicked.connect(self.process_image)
        layout.addWidget(select_button)

        # Set up the main window
        self.setWindowTitle('Porosity Detection App')
        self.setGeometry(100, 100, 800, 600)

        # Set the layout for the main window
        self.setLayout(layout)

    def process_image(self):
        # Open a file dialog to select an image
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.jpg *.bmp *.jpeg *.gif)')

        if file_path:
            # Load and preprocess the image
            img = Image.open(file_path)

            # Ensure the image has three channels (convert to RGB if needed)
            img = img.convert('RGB')

            # Perform model inference
            with torch.no_grad():
                output = self.model(img)

            # Process the API response
            bounding_boxes = []
            for result in output[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = result[:4]
                bounding_boxes.append([x1, y1, x2, y2])

            # Draw bounding boxes on the image
            img_with_boxes = self.draw_boxes_on_image(img, bounding_boxes)

            # Calculate aspect ratio and update the weights
            aspect_ratio = self.calculate_aspect_ratio(bounding_boxes)
            self.aspect_ratios.append(aspect_ratio)
            self.weights.append(len(bounding_boxes))  # Use the number of bounding boxes as weights

            # Calculate and display the weighted average aspect ratio
            weighted_avg_aspect_ratio = np.average(self.aspect_ratios, weights=self.weights)
            classification = self.classify_aspect_ratio(weighted_avg_aspect_ratio)

            # Display the image with bounding boxes and classification
            self.display_image_with_boxes(img_with_boxes, weighted_avg_aspect_ratio, classification)

    def calculate_aspect_ratio(self, bounding_boxes):
        # Assuming x1, y1, x2, y2 format for bounding boxes
        x1, y1, x2, y2 = bounding_boxes[0]
        width = x2 - x1
        height = y2 - y1

        aspect_ratio = width / height
        return aspect_ratio

    def draw_boxes_on_image(self, img, bounding_boxes):
        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(img)
        for box in bounding_boxes:
            x, y, x2, y2 = map(int, box)
            draw.rectangle([x, y, x2, y2], outline="red", width=2)

        return img

    def classify_aspect_ratio(self, aspect_ratio):
        if aspect_ratio == 1:
            return "Gas-out pores"
        elif 1 < aspect_ratio <= 2:
            return "Gas-out pores and Intergranular pores"
        elif 2 < aspect_ratio <= 3:
            return "Intergranular pores and Pull-Out Pores"
        elif 3 < aspect_ratio <= 4:
            return "Ejection pores"
        elif 4 < aspect_ratio <= 14:
            return "Capillaries"
    
    # Update the method to accept aspect_ratio and classification
    def display_image_with_boxes(self, img, aspect_ratio, classification):
        # Convert PIL image to QPixmap
        img_pixmap = QPixmap.fromImage(ImageQt(img))

        # Set the modified image with bounding boxes to the QLabel
        self.image_label.setPixmap(img_pixmap)

        # Display aspect ratio classification on the screen
        self.aspect_ratio_label.setText(f"Weighted Average Aspect Ratio: {aspect_ratio:.2f}, Classification: {classification}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    porosity_app = PorosityApp()
    porosity_app.show()
    sys.exit(app.exec_())
