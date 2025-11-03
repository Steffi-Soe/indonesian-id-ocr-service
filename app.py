import os
import json
from flask import Flask, request, jsonify, Response
from werkzeug.utils import secure_filename
from document_processor import DocumentProcessor
import uuid

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JSON_SORT_KEYS'] = False

print("Loading Document Processor...")
processor = DocumentProcessor()
print("Processor loaded. Flask server is ready.")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/ocr/document', methods=['POST'])
def process_document_image():
    if 'image' not in request.files:
        return jsonify({"status": 400, "error": True, "message": "Bad Request: 'image' part is missing in the form data"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"status": 400, "error": True, "message": "Bad Request: No file selected"}), 400

    if file and allowed_file(file.filename):
        _, file_extension = os.path.splitext(file.filename)
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(image_path)

        try:
            result = processor.process_image(image_path)
            
            status_code = result.get("status", 500)
            
            json_string = json.dumps(result, ensure_ascii=False, indent=4)
            return Response(json_string, status=status_code, content_type='application/json; charset=utf-8')

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"status": 500, "error": True, "message": f"An internal server error occurred: {e}"}), 500
        
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)
    else:
        return jsonify({"status": 400, "error": True, "message": f"Bad Request: File type not allowed. Please use one of {list(ALLOWED_EXTENSIONS)}"}), 400


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    from waitress import serve
    print("Starting server with Waitress...")
    serve(app, host='0.0.0.0', port=5000, threads=4)