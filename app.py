import os
import json
from flask import Flask, request, jsonify, Response
from document_processor import DocumentProcessor

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JSON_SORT_KEYS'] = False

print("Loading Document Processor...")
processor = DocumentProcessor()
print("Processor loaded. Flask server is ready.")

@app.route('/ocr/document', methods=['POST'])
def process_document_image():
    if not request.is_json:
        return jsonify({"status": 400, "error": True, "message": "Bad Request: Missing JSON body"}), 400

    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({"status": 400, "error": True, "message": "Bad Request: 'filename' key is missing"}), 400

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(image_path):
        return jsonify({"status": 404, "error": True, "message": f"File not found: {filename}"}), 404

    try:
        result = processor.process_image(image_path)
        
        status_code = result.get("status", 500)
        
        json_string = json.dumps(result, ensure_ascii=False, indent=4)
        return Response(json_string, status=status_code, content_type='application/json; charset=utf-8')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": 500, "error": True, "message": f"An internal server error occurred: {e}"}), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)