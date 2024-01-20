import sys
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

        self.init_ui()

    def init_ui(self):
        # Create layout
        layout = QVBoxLayout()

        # Create QLabel to display image
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

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
                x1, y1, x2, y2 = result[:4]  # Extracting the first four values (x1, y1, x2, y2)
                bounding_boxes.append([x1, y1, x2, y2])

            # Draw bounding boxes on the image
            img_with_boxes = self.draw_boxes_on_image(img, bounding_boxes)

            # Display the image with bounding boxes
            self.display_image_with_boxes(img_with_boxes)

    def draw_boxes_on_image(self, img, bounding_boxes):
        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(img)
        for box in bounding_boxes:
            x, y, x2, y2 = map(int, box)
            draw.rectangle([x, y, x2, y2], outline="red", width=2)

        return img

    def display_image_with_boxes(self, img):
        # Convert PIL image to QPixmap
        img_pixmap = QPixmap.fromImage(ImageQt(img))

        # Set the modified image with bounding boxes to the QLabel
        self.image_label.setPixmap(img_pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    porosity_app = PorosityApp()
    porosity_app.show()
    sys.exit(app.exec_())