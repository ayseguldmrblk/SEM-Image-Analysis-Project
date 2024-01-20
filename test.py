from PIL import Image, ImageDraw
from ultralytics import YOLO
import torch

class Analyzer:
    def __init__(self, model):
        self.model = model

    def process_image(self, image_path):
        # Load image
        img = Image.open(image_path)

        # Run inference
        with torch.no_grad():
            output = self.model(img)

        # Assuming output is a list and the first element contains the bounding boxes
        bounding_boxes = []
        for result in output[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, result[:4])
            bounding_boxes.append([x1, y1, x2 - x1, y2 - y1])

        # Save the image with bounding boxes
        self.save_image_with_boxes(img, bounding_boxes, output_path='output.jpg')

    def save_image_with_boxes(self, img, bounding_boxes, output_path):
        # Convert image to RGB mode
        img = img.convert('RGB')

        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(img)
        for box in bounding_boxes:
            x, y, width, height = map(int, box)
            draw.rectangle([x, y, x + width, y + height], outline="red", width=2)

        # Save the image
        img.save(output_path)

# Example usage
analyzer = Analyzer(model=YOLO('porosity_model.pt'))
analyzer.process_image('example.jpeg')
