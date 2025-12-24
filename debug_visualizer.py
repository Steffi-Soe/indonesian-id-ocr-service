import cv2
import json
import os
import numpy as np

class DebugVisualizer:
    def __init__(self, output_dir="debug_output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def visualize_ocr(self, image, ocr_data, prefix="ocr"):
        """
        Draws bounding boxes with IDs only (no text) to avoid clutter.
        Saves:
        1. Image with ID boxes.
        2. JSON 'Legend' mapping ID -> Text, Confidence, Box.
        """
        vis_image = image.copy()
        
        h, w = vis_image.shape[:2]
        font_scale = max(h, w) / 2000.0
        thickness = max(1, int(font_scale * 2))

        log_data = []

        for item in ocr_data:
            oid = item['id']
            box = np.array(item['box'], dtype=np.int32)
            
            cv2.polylines(vis_image, [box], True, (0, 255, 0), thickness)
            
            label = str(oid)
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            cv2.rectangle(vis_image, 
                          (box[0][0], box[0][1] - text_h - 5), 
                          (box[0][0] + text_w, box[0][1]), 
                          (0, 255, 0), -1)
            
            cv2.putText(vis_image, label, (box[0][0], box[0][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

            log_data.append({
                "id": oid,
                "text": item['text'],
                "confidence": float(item.get('confidence', 0.0)),
                "box": item['box'].tolist()
            })

        img_path = os.path.join(self.output_dir, f"{prefix}_visual.jpg")
        cv2.imwrite(img_path, vis_image)

        log_path = os.path.join(self.output_dir, f"{prefix}_legend.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        return img_path, log_path

    def save_extraction_trace(self, trace_data, prefix="extraction"):
        """
        Saves the traceability report explaining how fields were derived.
        """
        trace_path = os.path.join(self.output_dir, f"{prefix}_trace.json")
        
        summary = {
            "extracted_fields": trace_data,
            "statistics": {
                "total_fields_found": len(trace_data),
                "fields_missing": [k for k, v in trace_data.items() if v.get("status") == "missing"]
            }
        }
        
        with open(trace_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)