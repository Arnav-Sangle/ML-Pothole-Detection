from flask import Flask, render_template, request
from ultralytics import YOLO
import os
from PIL import Image, ImageDraw

app = Flask(__name__)



def draw_boxes(image_path, model_output):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for result in model_output:
        x1, y1, x2, y2, confidence, class_id = result.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"Class: {int(class_id)} - Confidence: {confidence}", fill="red")
    return image

# Function to process input and get model output
def process_input(image_file):
    # Save the uploaded image temporarily
    image_path = "temp_image.jpg"
    image_file.save(image_path)

    # Load a pretrained YOLOv8 model
    model = YOLO('C:/Users/ishas/Downloads/yolo8n-1280imgsz-50epoch/weights/best.pt')

    # Run inference on the saved image with arguments
    model_output = model.predict(image_path, save=False, imgsz=640, conf=0.5)

    # Draw bounding bo  xes on the image
    image_with_boxes = draw_boxes(image_path, model_output)

    # # Save the image with bounding boxes
    output_path = "output_image.jpg"
    image_with_boxes.save(output_path)

    # # Remove the temporary image file
    # os.remove(image_path)

    # return output_path
    return model_output


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image_input' not in request.files:
            return render_template('index.html', error="No image selected")

        image_file = request.files['image_input']  # Get the uploaded file
        if image_file.filename == '':
            return render_template('index.html', error="No image selected")

        model_output = process_input(image_file)  # Process input with the model
        return render_template('result.html', model_output=model_output)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
