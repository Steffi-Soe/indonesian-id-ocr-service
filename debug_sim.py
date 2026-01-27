import cv2
import numpy as np
import os
import json
import logging
from paddleocr import PaddleOCR
from sim_extractor import SIMExtractor, format_sim_to_json

IMAGE_PATH = "sim case/sim cam 5.jpg" 
OUTPUT_DIR = "debug_output"

logging.getLogger("ppocr").setLevel(logging.ERROR)

class DebugVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def draw_ocr_boxes(self, image, ocr_result, filename_prefix="debug"):
        vis_image = image.copy()
        
        if not ocr_result or not ocr_result[0]:
            print("[ERROR] No text detected.")
            return

        data = ocr_result[0]
        boxes = data.get('dt_polys', [])
        texts = data.get('rec_texts', [])

        print(f"\n--- RAW OCR DATA ({len(texts)} items) ---")
        print(f"{'ID':<4} | {'Y-Center':<8} | {'Text'}")
        print("-" * 50)

        for i, (box, text) in enumerate(zip(boxes, texts)):
            box = np.array(box).astype(np.int32)
            y_center = int((box[0][1] + box[2][1]) / 2)
            
            cv2.polylines(vis_image, [box], True, (0, 255, 0), 2)

            label = f"[{i}] {text}"
            origin = (box[0][0], box[0][1] - 10)
            
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_image, (origin[0], origin[1] - h), (origin[0] + w, origin[1]), (0, 0, 0), -1)
            cv2.putText(vis_image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            print(f"{i:<4} | {y_center:<8} | {text}")

        output_path = os.path.join(self.output_dir, f"{filename_prefix}_visualized.jpg")
        cv2.imwrite(output_path, vis_image)
        print(f"\n[INFO] Visualized image saved to: {output_path}")

def run_debug(image_path):
    print(f"Loading Image: {image_path}")
    if not os.path.exists(image_path):
        print("[ERROR] Image file does not exist.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] Failed to read image using cv2.")
        return

    print("Initializing OCR Engine...")
    ocr = PaddleOCR(use_textline_orientation=True, lang='id', enable_mkldnn=True)
    
    print("Running OCR...")
    result = ocr.predict(image)
    
    if not result or not result[0]:
        print("[ERROR] PaddleOCR returned no results.")
        return

    viz = DebugVisualizer(OUTPUT_DIR)
    viz.draw_ocr_boxes(image, result, filename_prefix="sim_debug")

    print("\n--- RUNNING EXTRACTOR ---")
    extractor = SIMExtractor()
    
    rec_texts = result[0]['rec_texts']

    raw_data = extractor.process_sim(result)
    json_output = format_sim_to_json(raw_data)

    print("\n--- FINAL JSON OUTPUT ---")
    print(json.dumps(json_output, indent=4))

    print("\n--- LOGIC DIAGNOSTICS ---")
    
    name = json_output['data']['nama']
    if name and ("MOBIL" in name or "PASSENGER" in name):
        print(f"[FAIL] Name Extraction failed. It picked up '{name}'.")
        print("Reason: This text is likely located geometrically to the right of the 'Nama' label.")
        print("Fix: Adjust 'find_value_to_right' max_x_dist or add 'MOBIL' to the ignore list in SIMExtractor.")
    elif not name:
        print("[FAIL] Name is NULL.")
    else:
        print(f"[PASS] Name seems okay: {name}")

    addr = json_output['data']['alamat']
    print(f"Address Captured: {addr}")
    if addr['kabupaten'] is None and addr['name'] is None:
        print("[FAIL] Address parsing failed entirely.")
        print("Reason: 'Alamat' anchor might be missed, or lines below it are too far away.")

if __name__ == "__main__":
    run_debug(IMAGE_PATH)