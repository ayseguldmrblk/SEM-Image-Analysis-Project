import sys
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt
import torch
from ultralytics import YOLO
from PyQt5.QtWidgets import QGridLayout, QGroupBox

class PorosityApp(QWidget):
    def __init__(self):
        super().__init__()

        # Load the PyTorch model
        self.model = YOLO('porosity_model.pt')

        # Initialize aspect ratios list and weights
        self.aspect_ratios = []
        self.weights = []  # Add this line to initialize the weights

        # Store file path and bounding boxes as class attributes
        self.file_path = None
        self.bounding_boxes = None

        self.init_ui()

    def init_ui(self):
        # Create layout
        layout = QGridLayout()

        # Create QGroupBox for image display
        image_group_box = QGroupBox('Image')
        image_layout = QVBoxLayout()
        self.image_label = QLabel(self)
        image_layout.addWidget(self.image_label)
        image_group_box.setLayout(image_layout)
        layout.addWidget(image_group_box, 0, 0, 2, 2)  # Span over two rows and two columns

        # Create QGroupBox for aspect ratio display
        aspect_ratio_group_box = QGroupBox('Aspect Ratio')
        aspect_ratio_layout = QVBoxLayout()
        self.aspect_ratio_label = QLabel(self)
        aspect_ratio_layout.addWidget(self.aspect_ratio_label)
        aspect_ratio_group_box.setLayout(aspect_ratio_layout)
        layout.addWidget(aspect_ratio_group_box, 0, 2, 1, 1)  # Span over one row and one column

        # Create QGroupBox for classification display
        classification_group_box = QGroupBox('Classification')
        classification_layout = QVBoxLayout()
        self.classification_label = QLabel(self)
        classification_layout.addWidget(self.classification_label)
        classification_group_box.setLayout(classification_layout)
        layout.addWidget(classification_group_box, 1, 2, 1, 1)  # Span over one row and one column

        # Create QPushButton for selecting an image
        select_button = QPushButton('Select Image', self)
        select_button.clicked.connect(self.process_image)
        layout.addWidget(select_button, 2, 0, 1, 3)  # Span over one row and three columns

        # Set up the main window
        self.setWindowTitle('Porosity Detection App')
        self.setGeometry(100, 100, 800, 600)

        # Set the layout for the main window
        self.setLayout(layout)

    def process_image(self):
        # Open a file dialog to select an image
        self.file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.jpg *.bmp *.jpeg *.gif)')

        if self.file_path:
            # Load and preprocess the image
            img = Image.open(self.file_path)

            # Ensure the image has three channels (convert to RGB if needed)
            img = img.convert('RGB')

            # Perform model inference
            with torch.no_grad():
                output = self.model(img)

            # Process the API response
            self.bounding_boxes = []

            # Assuming result is a Boxes object
            for result in output[0].boxes:
                # Extract bounding box information from the Boxes object
                xyxy = result.xyxy  # Access xyxy attribute

                # Extract confidence information using the conf property
                confidences = result.conf.tolist()

                # Convert xyxy tensor to a list
                self.bounding_boxes.extend(xyxy.cpu().numpy().tolist())

            # Draw bounding boxes on the image with confidence scores
            img_with_boxes = self.draw_boxes_on_image(img, self.bounding_boxes, confidences)

            # Calculate aspect ratio and update the weights
            aspect_ratio = self.calculate_aspect_ratio(self.bounding_boxes)
            self.aspect_ratios.append(aspect_ratio)
            self.weights.append(len(self.bounding_boxes))  # Use the number of bounding boxes as weights

            # Calculate and display the weighted average aspect ratio
            weighted_avg_aspect_ratio = np.average(self.aspect_ratios, weights=self.weights)
            classification = self.classify_aspect_ratio(weighted_avg_aspect_ratio)

            # Display the image with bounding boxes, classification, and confidence scores
            self.display_image_with_boxes(img_with_boxes, aspect_ratio, classification, confidences)

    def calculate_aspect_ratio(self, bounding_boxes):
        # Assuming x1, y1, x2, y2 format for bounding boxes
        x1, y1, x2, y2 = bounding_boxes[0]
        width = x2 - x1
        height = y2 - y1

        aspect_ratio = width / height
        return aspect_ratio

    def draw_boxes_on_image(self, img, bounding_boxes, confidences):
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

    def display_image_with_boxes(self, img, aspect_ratio, classification, confidences):
        # Convert PIL image to QPixmap
        img_pixmap = QPixmap.fromImage(ImageQt(img))

        # Set the modified image with bounding boxes to the QLabel
        self.image_label.setPixmap(img_pixmap)

        # Display the aspect ratio
        self.aspect_ratio_label.setText(f"Weighted Average Aspect Ratio: {aspect_ratio:.2f}")

        # Display the classification
        self.classification_label.setText(f"Classification: {classification}")

        # Save annotated image with bounding boxes
        result_dir = 'results'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # Add "_analysis" to the original file name
        file_name, file_extension = os.path.splitext(os.path.basename(self.file_path))
        result_file_path = os.path.join(result_dir, f"{file_name}_analysis{file_extension}")

        # Draw bounding boxes on the image with confidence scores
        img_with_boxes = self.draw_boxes_on_image(img.copy(), self.bounding_boxes, confidences)

        # Draw text at the top of the image
        draw = ImageDraw.Draw(img_with_boxes)
        text = f"Classification: {classification}\nAspect Ratio: {aspect_ratio:.2f}"
        text_position = (10, 10)  # Adjust the position as needed
        draw.text(text_position, text, fill="white")

        # Save the annotated image
        img_with_boxes.save(result_file_path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    porosity_app = PorosityApp()
    porosity_app.show()
    sys.exit(app.exec_())
