from flask import Flask, render_template, request, redirect, url_for, flash
from transformers import BlipProcessor, BlipForConditionalGeneration
import PIL
from PIL import Image
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = 'D:\Generative_AI_Introduction_and_Applications\Image_Captioning\Images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the BLIP model and processor once
processor, model = None, None

def load_model_and_processor():
    global processor, model
    if processor is None or model is None:
        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            print("Model and processor loaded successfully!")
        except Exception as e:
            print(f"Error loading model or processor: {e}")

def validate_image(image_path: str) -> bool:
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return False
    try:
        image = Image.open(image_path)
        return isinstance(image, Image.Image)
    except PIL.UnidentifiedImageError:
        print(f"Invalid image file: {image_path}")
        return False

def generate_caption(image: Image.Image, processor, model, max_length: int = 50) -> str:
    try:
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=max_length)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"An error occurred during caption generation: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Validate and generate the caption
        if validate_image(filepath):
            image = Image.open(filepath)
            caption = generate_caption(image, processor, model)
            return render_template('index.html', uploaded_image=file.filename, caption=caption)
        else:
            flash('Invalid image file')
            return redirect(url_for('index'))

if __name__ == '__main__':
    load_model_and_processor()
    app.run(debug=True)
