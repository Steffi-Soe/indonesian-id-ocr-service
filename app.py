import os
import json
from flask import Flask, request, jsonify, Response
from ktp_extractor import KTPExtractor, format_to_target_json

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JSON_SORT_KEYS'] = False

print("Loading KTP Extractor model...")
extractor = KTPExtractor()
print("Model loaded. Flask server is ready.")

@app.route('/ocr/ktp', methods=['POST'])
def process_ktp_image():
    """
    This endpoint receives a JSON payload with a filename,
    processes the corresponding image, and returns the extracted data.
    """
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
        ktp_data = extractor.process_ktp(image_path)
        
        if not ktp_data:
            return jsonify({"status": 500, "error": True, "message": "Could not extract data from the image."}), 500

        formatted_response_dict = format_to_target_json(ktp_data)
        json_string = json.dumps(formatted_response_dict, ensure_ascii=False, indent=4)
        
        return Response(json_string, content_type='application/json; charset=utf-8')

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"status": 500, "error": True, "message": f"An internal server error occurred: {e}"}), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)