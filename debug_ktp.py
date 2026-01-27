import cv2
import numpy as np
import os
import json
import logging
from paddleocr import PaddleOCR
from ktp_extractor import KTPExtractor, format_to_target_json
from image_preprocessor import ImagePreprocessor

# --- CONFIGURATION ---
IMAGE_PATH = "ktp case/ktp cam 2.jpg"  # Corrected path based on your log
OUTPUT_DIR = "debug_output_ktp"

# Suppress PaddleOCR internal logging
logging.getLogger("ppocr").setLevel(logging.ERROR)

class DebugVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def draw_ocr_boxes(self, image, ocr_result, filename_prefix="ktp_debug"):
        vis_image = image.copy()
        
        if not ocr_result or not ocr_result[0]:
            print("[ERROR] No text detected.")
            return

        data = ocr_result[0]
        boxes = data.get('dt_polys', [])
        texts = data.get('rec_texts', [])
        scores = data.get('rec_scores', [])

        print(f"\n--- RAW OCR DATA ({len(texts)} items) ---")
        print(f"{'ID':<4} | {'Conf':<6} | {'Y-Center':<8} | {'Text'}")
        print("-" * 60)

        for i, (box, text, score) in enumerate(zip(boxes, texts, scores)):
            box = np.array(box).astype(np.int32)
            y_center = int((box[0][1] + box[2][1]) / 2)
            
            # Color code: Green = High Conf, Red = Low Conf
            color = (0, 255, 0) if score > 0.8 else (0, 0, 255)
            
            cv2.polylines(vis_image, [box], True, color, 2)

            label = f"[{i}] {text}"
            origin = (box[0][0], box[0][1] - 10)
            
            # Draw text background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (origin[0], origin[1] - h), (origin[0] + w, origin[1]), (0, 0, 0), -1)
            cv2.putText(vis_image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            print(f"{i:<4} | {score:.2f}   | {y_center:<8} | {text}")

        output_path = os.path.join(self.output_dir, f"{filename_prefix}_visualized.jpg")
        cv2.imwrite(output_path, vis_image)
        print(f"\n[INFO] Visualized image saved to: {output_path}")

def run_debug_ktp(image_path):
    print(f"Loading Image: {image_path}")
    if not os.path.exists(image_path):
        print("[ERROR] Image file does not exist.")
        return

    raw_image = cv2.imread(image_path)
    if raw_image is None:
        print("[ERROR] Failed to read image using cv2.")
        return

    # 1. PREPROCESSING
    print("\n--- RUNNING PREPROCESSOR ---")
    preprocessor = ImagePreprocessor(debug=True, debug_dir=os.path.join(OUTPUT_DIR, "preprocess"))
    processed_image = preprocessor.preprocess(raw_image)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "final_input_to_ocr.jpg"), processed_image)

    # 2. OCR
    print("\n--- RUNNING PADDLEOCR ---")
    # FIX: Replaced use_angle_cls with use_textline_orientation and removed show_log
    ocr = PaddleOCR(use_textline_orientation=True, lang='id', enable_mkldnn=True)
    
    result = ocr.predict(processed_image)
    
    if not result or not result[0]:
        print("[ERROR] PaddleOCR returned no results.")
        return

    # 3. VISUALIZATION
    viz = DebugVisualizer(OUTPUT_DIR)
    viz.draw_ocr_boxes(processed_image, result, filename_prefix="ktp_ocr_result")

    # 4. EXTRACTION
    print("\n--- RUNNING KTP EXTRACTOR ---")
    extractor = KTPExtractor()
    
    cleaned_data, raw_filtered, trace_info = extractor.process_ktp(result, return_trace=True)
    json_output = format_to_target_json(cleaned_data)

    # 5. OUTPUT DISPLAY
    print("\n--- EXTRACTOR TRACE INFO (How it found the fields) ---")
    if trace_info:
        for key, info in trace_info.items():
            print(f"[{key:<18}] Val: {info['value']:<25} | IDs: {info['source_ids']} | Method: {info['method']}")
    else:
        print("No trace info returned (Extraction likely failed).")

    print("\n--- FINAL JSON OUTPUT ---")
    print(json.dumps(json_output, indent=4))

    # 6. LOGIC DIAGNOSTICS
    print("\n--- LOGIC DIAGNOSTICS ---")
    data = json_output.get('data', {})
    
    # Check NIK
    nik = data.get('nomor')
    if nik and len(nik) >= 15 and nik.isdigit():
        print(f"[PASS] NIK is valid format: {nik}")
    else:
        print(f"[WARN] NIK might be invalid or incomplete: {nik}")

    # Check Name
    nama = data.get('nama')
    if nama and not any(char.isdigit() for char in nama):
        print(f"[PASS] Name extracted: {nama}")
    else:
        print(f"[FAIL] Name missing or contains numbers: {nama}")

    # Check Address
    addr = data.get('alamat', {})
    if addr.get('provinsi') and addr.get('kabupaten'):
        print(f"[PASS] Headers (Prov/Kab) found: {addr.get('provinsi')} - {addr.get('kabupaten')}")
    else:
        print("[FAIL] Headers missing. Ensure Top-Down filtering isn't cutting them off.")

if __name__ == "__main__":
    run_debug_ktp(IMAGE_PATH)