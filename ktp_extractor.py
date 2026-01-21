import re
import numpy as np
from thefuzz import process, fuzz


class KTPExtractor:
    def __init__(self):
        self.canonical_fields = [
            "PROVINSI", "KABUPATEN", "NIK", "Nama", "Tempat/Tgl Lahir",
            "Jenis Kelamin", "Gol. Darah", "Alamat", "RT/RW", "Kel/Desa",
            "Kecamatan", "Agama", "Status Perkawinan", "Pekerjaan",
            "Kewarganegaraan", "Berlaku Hingga"
        ]

        self.truncated_key_map = {
            # "DAMAT": "Alamat",
            "NIS KELAMIN": "Jenis Kelamin",
            "ENIS KELAMIN": "Jenis Kelamin",
            "EMPAT/TGL": "Tempat/Tgl Lahir",
            "MPAT/TGL": "Tempat/Tgl Lahir",
            "GAMA": "Agama",
            "KERJAAN": "Pekerjaan",
            "ATUS PERKAWINAN": "Status Perkawinan",
            "KAL/DESA": "Kel/Desa",
            "KEL/DESA": "Kel/Desa"
        }

        self.known_values = {
            "Agama": [
                "ISLAM", "KRISTEN", "KATOLIK", "HINDU", "BUDDHA", "KONGHUCU",
                "CHRISTIAN", "CATHOLIC" # WNA support
            ],
            "Jenis Kelamin": [
                "LAKI-LAKI", "PEREMPUAN", "LAKI", "PEREMPUAN",
                "MALE", "FEMALE" # WNA support
            ],
            "Status Perkawinan": [
                "BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI",
                "MARRIED", "SINGLE", "DIVORCED" # WNA support
            ],
            "Kewarganegaraan": ["WNI", "WNA"]
        }

    def _get_y_center(self, item):
        box = item['box']
        return (box[0][1] + box[3][1]) / 2

    def process_ktp(self, ocr_result, return_trace=False):
        if not ocr_result or not ocr_result[0]:
            return (None, None, None) if return_trace else None

        result_dict = ocr_result[0]
        if not result_dict or not isinstance(result_dict, dict):
            return (None, None, None) if return_trace else None

        boxes = result_dict.get('dt_polys', [])
        texts = result_dict.get('rec_texts', [])
        scores = result_dict.get('rec_scores', [])

        if not texts:
            return (None, None, None) if return_trace else None

        recognized_data = []
        for i, (box, text) in enumerate(zip(boxes, texts)):
            conf = scores[i] if i < len(scores) else 0.0
            recognized_data.append({
                'id': i,
                'box': np.array(box).astype(np.int32),
                'text': text,
                'confidence': conf
            })

        filtered_data = self.filter_spatial_outliers(recognized_data)

        structured_data, trace_info = self.post_process(filtered_data)

        cleaned_data = self.cleanup_data(structured_data)

        if return_trace:
            return cleaned_data, filtered_data, trace_info
        return cleaned_data

    def filter_spatial_outliers(self, recognized_data):
        key_y_positions = []
        for item in recognized_data:
            text_upper = item['text'].upper()
            match, score = process.extractOne(
                text_upper, self.canonical_fields, scorer=fuzz.partial_ratio
            )
            if score > 85:
                key_y_positions.append(self._get_y_center(item))

        if not key_y_positions:
            return recognized_data

        min_y = min(key_y_positions)
        max_y = max(key_y_positions)

        active_height = max_y - min_y
        cutoff_y = max_y + (active_height * 0.45)

        filtered = [
            item for item in recognized_data
            if self._get_y_center(item) <= cutoff_y
        ]
        return filtered

    def post_process(self, recognized_data):
        potential_keys = []
        potential_values = []
        trace_info = {}

        for item in recognized_data:
            text_raw = item['text'].strip()
            text_upper = text_raw.upper()

            if len(text_raw) < 2 and text_raw not in [":", "-"]:
                potential_values.append(item)
                continue

            best_match, score = process.extractOne(
                text_raw, self.canonical_fields, scorer=fuzz.partial_ratio
            )

            truncated_match = None
            for bad_key, correct_key in self.truncated_key_map.items():
                if bad_key in text_upper:
                    truncated_match = correct_key
                    break

            is_key = False

            if truncated_match:
                item['canonical_field'] = truncated_match
                potential_keys.append(item)
                is_key = True
            elif score > 80:
                item['canonical_field'] = best_match
                potential_keys.append(item)
                is_key = True

            if not is_key:
                potential_values.append(item)

        potential_keys.sort(key=self._get_y_center)
        key_map = {k['canonical_field']: k for k in potential_keys}

        extracted_data = {}
        claimed_value_ids = set()

        for key_item in potential_keys:
            key_name = key_item['canonical_field']

            if key_name in extracted_data:
                continue

            if key_name in ["PROVINSI", "KABUPATEN"]:
                value = re.sub(
                    re.escape(key_name), '', key_item['text'],
                    flags=re.IGNORECASE
                ).strip()

                value = re.sub(r'^[:\-\.\s]+', '', value).strip()

                if value:
                    extracted_data[key_name] = value
                    trace_info[key_name] = {
                        "value": value,
                        "source_ids": [key_item['id']],
                        "method": "header_strip"
                    }
                    continue

            value_parts = re.split(r':\s*', key_item['text'], maxsplit=1)
            if len(value_parts) > 1 and value_parts[1].strip():
                val = value_parts[1].strip()
                extracted_data[key_name] = val
                trace_info[key_name] = {
                    "value": val,
                    "source_ids": [key_item['id']],
                    "method": "inline_regex_split"
                }
                continue

            key_y_center = self._get_y_center(key_item)
            key_x_end = key_item['box'][1][0]
            same_line_candidates = []

            vertical_threshold = 25 

            for val_item in potential_values:
                if val_item['id'] in claimed_value_ids:
                    continue

                val_y_center = self._get_y_center(val_item)

                if (abs(val_y_center - key_y_center) < vertical_threshold and
                        val_item['box'][0][0] > key_x_end):
                    
                    x_dist = val_item['box'][0][0] - key_x_end
                    y_diff = abs(val_y_center - key_y_center)
                    
                    score = x_dist + (y_diff * 15)
                    
                    same_line_candidates.append((score, val_item))

            if same_line_candidates:
                same_line_candidates.sort(key=lambda c: c[0])

                valid_candidates = [
                    c for c in same_line_candidates
                    if not re.match(r'^[:\-\.\s]+$', c[1]['text'])
                ]

                if valid_candidates:
                    best_candidate = valid_candidates[0][1]
                    value_text = best_candidate['text']
                    used_ids = [best_candidate['id']]
                    method = "geometric_match"

                    if key_name == 'Alamat':
                        rt_rw_key = key_map.get('RT/RW')
                        rt_rw_y = (
                            self._get_y_center(rt_rw_key)
                            if rt_rw_key else float('inf')
                        )
                        addr_line1_y = self._get_y_center(best_candidate)
                        
                        second_line_cands = []
                        for val_item in potential_values:
                            if val_item['id'] in claimed_value_ids:
                                continue
                            if val_item['id'] == best_candidate['id']:
                                continue

                            val_y = self._get_y_center(val_item)
                            
                            is_below_addr = val_y > (addr_line1_y + 10)
                            is_above_rtrw = val_y < (rt_rw_y - 10)
                            
                            is_close = (val_y - addr_line1_y) < 45

                            if is_below_addr and is_above_rtrw and is_close:
                                second_line_cands.append(val_item)

                        if second_line_cands:
                            second_line_cands.sort(key=lambda c: c['box'][0][1])
                            second_line = second_line_cands[0]
                            value_text += f" {second_line['text']}"
                            claimed_value_ids.add(second_line['id'])
                            used_ids.append(second_line['id'])
                            method = "geometric_match_multiline"

                    extracted_data[key_name] = value_text
                    claimed_value_ids.add(best_candidate['id'])

                    trace_info[key_name] = {
                        "value": value_text,
                        "source_ids": used_ids,
                        "key_id_used": key_item['id'],
                        "method": method
                    }
            
            if key_name == "NIK" and key_name not in extracted_data:
                below_candidates = []
                for val_item in potential_values:
                    if val_item['id'] in claimed_value_ids:
                        continue
                    
                    val_y_center = self._get_y_center(val_item)
                    y_diff = val_y_center - key_y_center
                    
                    if 0 < y_diff < 50:
                        clean_val = val_item['text'].replace(" ", "").replace(":","")
                        if re.match(r'\d+', clean_val):
                            below_candidates.append(val_item)
                
                if below_candidates:
                    below_candidates.sort(key=lambda x: x['box'][0][1])
                    best_nik = below_candidates[0]
                    extracted_data["NIK"] = best_nik['text']
                    claimed_value_ids.add(best_nik['id'])
                    trace_info["NIK"] = {
                        "value": best_nik['text'],
                        "source_ids": [best_nik['id']],
                        "method": "geometric_below_fallback"
                    }

        self.recover_missing_fields(
            extracted_data, potential_values, claimed_value_ids,
            key_map, trace_info
        )

        return {
            field: extracted_data.get(field)
            for field in self.canonical_fields if extracted_data.get(field)
        }, trace_info

    def recover_missing_fields(self, extracted, values, claimed_ids,
                               key_map, trace_info):

        for field, keywords in self.known_values.items():
            if field in extracted:
                continue

            for val_item in values:
                if val_item['id'] in claimed_ids:
                    continue

                text_upper = val_item['text'].upper()
                match = process.extractOne(
                    text_upper, keywords, scorer=fuzz.token_set_ratio
                )

                if not match and field == "Jenis Kelamin":
                    if "LAKILAKI" in text_upper:
                        extracted[field] = "LAKI-LAKI"
                        claimed_ids.add(val_item['id'])
                        trace_info[field] = {
                            "value": "LAKI-LAKI",
                            "source_ids": [val_item['id']],
                            "method": "typo_recovery"
                        }
                        continue

                if match and match[1] > 85:
                    extracted[field] = val_item['text']
                    claimed_ids.add(val_item['id'])
                    trace_info[field] = {
                        "value": val_item['text'],
                        "source_ids": [val_item['id']],
                        "method": "value_keyword_recovery"
                    }
                    break

        if "Nama" not in extracted:
            nik_key = key_map.get("NIK")
            ttl_key = key_map.get("Tempat/Tgl Lahir")

            y_min = -1
            y_max = float('inf')

            if nik_key:
                y_min = nik_key['box'][3][1]
            elif "NIK" in extracted and "NIK" in trace_info:
                 pass 

            if ttl_key:
                y_max = ttl_key['box'][0][1]

            candidates = []
            for val_item in values:
                if val_item['id'] in claimed_ids:
                    continue

                y_center = self._get_y_center(val_item)

                valid = False
                if y_min != -1 and y_max != float('inf'):
                    if y_min < y_center < y_max:
                        valid = True
                elif y_min != -1:
                    if y_min < y_center < y_min + 70:
                        valid = True
                elif y_max != float('inf'):
                    if y_max - 70 < y_center < y_max:
                        valid = True

                if valid:
                    candidates.append(val_item)

            if candidates:
                candidates.sort(key=lambda c: c['box'][0][0])
                chosen = candidates[0]
                extracted["Nama"] = chosen['text']
                claimed_ids.add(chosen['id'])
                trace_info["Nama"] = {
                    "value": chosen['text'],
                    "source_ids": [chosen['id']],
                    "method": "positional_inference_name"
                }
        
        if "NIK" not in extracted:
            for val_item in values:
                if val_item['id'] in claimed_ids:
                    continue
                clean_text = val_item['text'].replace(" ", "").strip()
                if re.match(r'^\d{16}$', clean_text):
                     extracted["NIK"] = clean_text
                     claimed_ids.add(val_item['id'])
                     trace_info["NIK"] = {
                        "value": clean_text,
                        "source_ids": [val_item['id']],
                        "method": "regex_recovery_16_digits"
                    }
                     break

    def cleanup_data(self, data):
        cleaned_data = {}
        for key, value in data.items():
            if value is None:
                continue

            clean_value = value.strip()
            if clean_value.startswith(':'):
                clean_value = clean_value[1:].strip()

            if key not in ["Tempat/Tgl Lahir", "Berlaku Hingga"]:
                clean_value = re.sub(
                    r'\s+\d{2}-\d{2}-\d{4}$', '', clean_value
                ).strip()

            if key == "Jenis Kelamin":
                val_upper = clean_value.upper()
                if "LAKI" in val_upper or "MALE" in val_upper:
                    clean_value = "LAKI-LAKI"
                elif "PEREMPUAN" in val_upper or "FEMALE" in val_upper:
                    clean_value = "PEREMPUAN"
                else:
                    if "LAK" in val_upper or "LK" in val_upper:
                        clean_value = "LAKI-LAKI"
                    elif "PER" in val_upper or "PR" in val_upper:
                        clean_value = "PEREMPUAN"

            if key == "Status Perkawinan":
                val_upper = clean_value.upper()
                if any(k in val_upper for k in ["BELUM", "SINGLE"]):
                    clean_value = "BELUM KAWIN"
                elif any(k in val_upper for k in ["KAWIN", "MARRIED"]):
                    clean_value = "KAWIN"
                elif "CERAI" in val_upper or "DIVORCED" in val_upper:
                    if "HIDUP" in val_upper:
                        clean_value = "CERAI HIDUP"
                    elif "MATI" in val_upper:
                        clean_value = "CERAI MATI"
                    else:
                        clean_value = "CERAI"

            if key == "Alamat":
                clean_value = re.sub(
                    r'\s+RT.*', '', clean_value, flags=re.IGNORECASE
                ).strip()
                clean_value = re.sub(
                    r'\s+RW.*', '', clean_value, flags=re.IGNORECASE
                ).strip()

            if key == "Pekerjaan":
                clean_value = clean_value.replace("BURUHHARIAN", "BURUH HARIAN")

            cleaned_data[key] = clean_value

        return cleaned_data


def format_to_target_json(data):
    tempat_lahir = None
    tgl_lahir = None

    raw_ttl = data.get("Tempat/Tgl Lahir", "")

    if raw_ttl:
        match = re.search(r'(.*?)[,.\s]+(\d{2}-\d{2}-\d{4})', raw_ttl)

        if match:
            tempat_lahir = match.group(1).strip().strip(":.,")
            tgl_lahir = match.group(2).strip()
        else:
            parts = raw_ttl.split(',', 1)
            tempat_lahir = parts[0].strip().strip(":.,")
            if len(parts) > 1:
                tgl_lahir = parts[1].strip()

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