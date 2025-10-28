import cv2
import re
import json
import numpy as np
from paddleocr import PaddleOCR
import os
from thefuzz import process, fuzz

class KTPExtractor:
    def __init__(self):
        # print("Initializing PaddleOCR engine...")
        # self.ocr = PaddleOCR(use_textline_orientation=True, lang='id')
        
        self.canonical_fields = [
            "PROVINSI", "KABUPATEN", "NIK", "Nama", "Tempat/Tgl Lahir",
            "Jenis Kelamin", "Gol. Darah", "Alamat", "RT/RW", "Kel/Desa",
            "Kecamatan", "Agama", "Status Perkawinan", "Pekerjaan",
            "Kewarganegaraan", "Berlaku Hingga"
        ]

    def _get_y_center(self, item):
        box = item['box']
        return (box[0][1] + box[3][1]) / 2

    def process_ktp(self, ocr_result):
        if not ocr_result or not ocr_result[0]:
            return None
        
        result_dict = ocr_result[0]
        if not result_dict or not isinstance(result_dict, dict):
             return None

        boxes = result_dict.get('dt_polys', [])
        texts = result_dict.get('rec_texts', [])
        
        if not texts:
            return None

        recognized_data = []
        for i, (box, text) in enumerate(zip(boxes, texts)):
            recognized_data.append({
                'id': i,
                'box': np.array(box).astype(np.int32),
                'text': text
            })

        structured_data = self.post_process(recognized_data)
        cleaned_data = self.cleanup_data(structured_data)
        return cleaned_data

    def post_process(self, recognized_data):
        potential_keys, potential_values = [], []
        for item in recognized_data:
            best_match, score = process.extractOne(item['text'], self.canonical_fields, scorer=fuzz.partial_ratio)
            if score > 85:
                item['canonical_field'] = best_match
                potential_keys.append(item)
            else:
                potential_values.append(item)

        potential_keys.sort(key=self._get_y_center)
        key_map = {k['canonical_field']: k for k in potential_keys}
        extracted_data = {}
        claimed_value_ids = set()

        for key_item in potential_keys:
            key_name = key_item['canonical_field']
            if key_name in extracted_data: continue

            if key_name in ["PROVINSI", "KABUPATEN"]:
                value = re.sub(re.escape(key_name), '', key_item['text'], flags=re.IGNORECASE).strip()
                if value:
                    extracted_data[key_name] = value
                    continue

            value_parts = re.split(r':\s*', key_item['text'], maxsplit=1)
            if len(value_parts) > 1 and value_parts[1].strip():
                extracted_data[key_name] = value_parts[1].strip()
                continue

            key_y_center = self._get_y_center(key_item)
            key_x_end = key_item['box'][1][0]
            
            same_line_candidates = []
            for val_item in potential_values:
                if val_item['id'] in claimed_value_ids: continue
                val_y_center = self._get_y_center(val_item)
                if abs(val_y_center - key_y_center) < 15 and val_item['box'][0][0] > key_x_end:
                    distance = val_item['box'][0][0] - key_x_end
                    same_line_candidates.append((distance, val_item))

            if same_line_candidates:
                same_line_candidates.sort(key=lambda c: c[0])
                best_candidate = same_line_candidates[0][1]
                value_text = best_candidate['text']
                
                if key_name == 'Alamat':
                    rt_rw_key = key_map.get('RT/RW')
                    rt_rw_y_center = self._get_y_center(rt_rw_key) if rt_rw_key else float('inf')
                    addr_line1_y = self._get_y_center(best_candidate)
                    second_line_candidates = []
                    for val_item in potential_values:
                        if val_item['id'] in claimed_value_ids: continue
                        val_y = self._get_y_center(val_item)
                        if (val_y > addr_line1_y and (val_y - addr_line1_y) < 35 and
                            abs(val_y - rt_rw_y_center) > 15):
                            second_line_candidates.append(val_item)
                    if second_line_candidates:
                        second_line_candidates.sort(key=lambda c: c['box'][0][1])
                        second_line = second_line_candidates[0]
                        value_text += f" {second_line['text']}"
                        claimed_value_ids.add(second_line['id'])
                extracted_data[key_name] = value_text
                claimed_value_ids.add(best_candidate['id'])
        return {field: extracted_data.get(field) for field in self.canonical_fields if extracted_data.get(field)}

    def cleanup_data(self, data):
        cleaned_data = {}
        for key, value in data.items():
            if value is None: continue
            clean_value = value.strip().replace(':', '').strip()
            if key not in ["Tempat/Tgl Lahir", "Berlaku Hingga"]:
                clean_value = re.sub(r'\s+\d{2}-\d{2}-\d{4}$', '', clean_value).strip()
            if key == "Jenis Kelamin":
                val_upper = clean_value.upper()
                if any(s in val_upper for s in ["LAKI", "MAE", "MALE"]): clean_value = "LAKI-LAKI"
                elif "PEREMPUAN" in val_upper: clean_value = "PEREMPUAN"
            if key == "Status Perkawinan":
                val_upper = clean_value.upper()
                if "BELUM" in val_upper: clean_value = "BELUM KAWIN"
                elif any(s in val_upper for s in ["KAWIN", "MARRIED"]): clean_value = "KAWIN"
            cleaned_data[key] = clean_value
        return cleaned_data

def format_to_target_json(data):
    tempat_lahir, tgl_lahir = None, None
    if data.get("Tempat/Tgl Lahir"):
        parts = data.get("Tempat/Tgl Lahir", "").split(',', 1)
        tempat_lahir = parts[0].strip() if len(parts) > 0 else None
        tgl_lahir = parts[1].strip() if len(parts) > 1 else None
    
    return {
        "status": 200,
        "error": False,
        "message": "KTP OCR Processed Successfully",
        "data": {
            "document_type": "KTP",
            "nomor": data.get("NIK"),
            "nama": data.get("Nama"),
            "tempat_lahir": tempat_lahir,
            "tgl_lahir": tgl_lahir,
            "jenis_kelamin": data.get("Jenis Kelamin"),
            "agama": data.get("Agama"),
            "status_perkawinan": data.get("Status Perkawinan"),
            "pekerjaan": data.get("Pekerjaan"),
            "kewarganegaraan": data.get("Kewarganegaraan"),
            "alamat": {
                "name": data.get("Alamat"),
                "rt_rw": data.get("RT/RW"),
                "kel_desa": data.get("Kel/Desa"),
                "kecamatan": data.get("Kecamatan"),
                "kabupaten": data.get("KABUPATEN"),
                "provinsi": data.get("PROVINSI")
            },
        }
    }