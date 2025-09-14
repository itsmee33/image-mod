from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import uuid
import requests
from PIL import Image
from transformers import pipeline
import easyocr
import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageModerator:
    def __init__(self):
        print("Initializing the Image Moderator... This may take a moment as models are downloaded.")
        
        print("Loading Zero-Shot classification model...")
        self.classifier = pipeline('zero-shot-image-classification', model='openai/clip-vit-large-patch14')
       
        self.CLASSIFICATION_LABELS = ['safe content', 'suggestive content', 'explicit nudity', 'graphic violence']
      
        print("Loading Object detection model...")
        self.object_detector = pipeline('object-detection', model='hustvl/yolos-tiny')
        self.WEAPON_OBJECTS = {'gun', 'knife', 'sword', 'weapon', 'revolver', 'pistol'}

        print("Loading Sensitive Symbol/Text OCR model...")
        self.ocr_reader = easyocr.Reader(['en'])
        self.HATE_KEYWORDS = {'hate', 'nazi', 'swastika'}

        print("Initialization complete. The moderator is ready.\n")

    def moderate_image(self, image_path):
        print(f"--- Moderating Image: {image_path} ---")
        try:
            if image_path.startswith('http'):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB")
        except Exception as e:
            return {"error": f"Failed to load image. {e}"}

        # --- Run all three models ---
        classification_results = self._classify_content(image)
        weapon_results = self._detect_weapons(image)
        hate_symbol_results = self._detect_hate_symbols(image_path)

        # Combine classification and weapon results into a single category
        violence_graphic_content_report = {
            "is_graphic_violence": classification_results["scores"]['graphic violence'] > "70.00%", # Flag if high probability
            "graphic_violence_score": classification_results["scores"]['graphic violence'],
            "contains_weapons": weapon_results["contains_weapons"],
            "detected_weapons": weapon_results["detected_items"]
        }

        report = {
            "adult_content_detection": {
                "is_nsfw": classification_results["is_nsfw"],
                "scores": {
                    "safe": classification_results["scores"]['safe content'],
                    "suggestive": classification_results["scores"]['suggestive content'],
                    "explicit": classification_results["scores"]['explicit nudity']
                }
            },
            
            "violence_graphic_content_detection": violence_graphic_content_report,
            "hate_symbol_detection": hate_symbol_results
        }
        return report

    def _classify_content(self, image):
        predictions = self.classifier(image, candidate_labels=self.CLASSIFICATION_LABELS)
        scores = {p['label']: f"{p['score']:.2%}" for p in predictions}
        
        
        is_nsfw_flag = scores['suggestive content'] > "70.00%" or scores['explicit nudity'] > "70.00%"
        
        return {
            "is_nsfw": is_nsfw_flag, 
            "scores": scores
        }

    def _detect_weapons(self, image):
        detected_objects = self.object_detector(image)
        found_weapons = [
            {"object": obj['label'], "confidence": f"{obj['score']:.2%}"}
            for obj in detected_objects
            if any(vo in obj['label'] for vo in self.WEAPON_OBJECTS)
        ]
        return {
            "contains_weapons": len(found_weapons) > 0,
            "detected_items": found_weapons
        }

    def _detect_hate_symbols(self, image_path):
        ocr_results = self.ocr_reader.readtext(image_path, detail=0, paragraph=True)
        detected_keywords = [
            {"keyword_found": keyword, "context": text_block}
            for text_block in ocr_results
            for keyword in self.HATE_KEYWORDS
            if keyword in text_block.lower()
        ]
        return {
            "contains_hate_keywords": len(detected_keywords) > 0,
            "detected_text": detected_keywords
        }



app = Flask(__name__)
CORS(app)

print("Initializing ImageModerator. This will happen only once at server startup.")
moderator = ImageModerator()
print("ImageModerator is ready to receive requests.")

uploads_dir = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

@app.route('/')
def index():
    try:
        with open('moderation_ui.html', 'r', encoding='utf-8') as f:
            return render_template_string(f.read())
    except FileNotFoundError:
        return "Error: moderation_ui.html not found. Please make sure it's in the same directory as app.py.", 404

@app.route('/moderate', methods=['POST'])
def moderate_endpoint():
    # (This part remains unchanged)
    if 'image_file' in request.files:
        image_file = request.files['image_file']
        if image_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        extension = os.path.splitext(image_file.filename)[1]
        temp_filename = str(uuid.uuid4()) + extension
        temp_filepath = os.path.join(uploads_dir, temp_filename)
        image_file.save(temp_filepath)
        
        report = moderator.moderate_image(temp_filepath)
        
        os.remove(temp_filepath)
        return jsonify(report)
     
    elif request.json and 'image_url' in request.json:
        image_url = request.json.get('image_url')
        if not image_url:
            return jsonify({"error": "Image URL is empty"}), 400
        report = moderator.moderate_image(image_url)
        return jsonify(report)
    else:
        return jsonify({"error": "Invalid request. Provide either 'image_file' or 'image_url'."}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
