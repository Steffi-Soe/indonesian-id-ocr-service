import re
import numpy as np


class GeometryUtils:

    @staticmethod
    def calculate_iou(box1, box2):
        x1 = max(box1[0][0], box2[0][0])
        y1 = max(box1[0][1], box2[0][1])
        x2 = min(box1[2][0], box2[2][0])
        y2 = min(box1[2][1], box2[2][1])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2][0] - box1[0][0]) * (box1[2][1] - box1[0][1])
        box2_area = (box2[2][0] - box2[0][0]) * (box2[2][1] - box2[0][1])

        denominator = float(box1_area + box2_area - inter_area)
        return inter_area / denominator if denominator > 0 else 0

    @staticmethod
    def find_value_to_right(anchor_text, data_list, y_thresh=20, max_x=500):
        anchor = next(
            (x for x in data_list if anchor_text.lower() in x['text'].lower()),
            None
        )
        if not anchor:
            return None

        anchor_y = anchor['y_center']
        anchor_right = anchor['box'][1][0]

        candidates = []
        for item in data_list:
            if item == anchor:
                continue
            if abs(item['y_center'] - anchor_y) < y_thresh:
                dist = item['box'][0][0] - anchor_right
                if 0 < dist < max_x:
                    candidates.append(item)

        candidates.sort(key=lambda x: x['box'][0][0])
        return " ".join([c['text'] for c in candidates]) if candidates else None

    @staticmethod
    def find_value_below(anchor_text, data_list, lookahead=5, x_var=600):
        anchor = next(
            (x for x in data_list if anchor_text.lower() in x['text'].lower()),
            None
        )
        if not anchor:
            return None

        anchor_y_bottom = anchor['box'][2][1]
        anchor_x_center = (anchor['box'][0][0] + anchor['box'][1][0]) / 2

        candidates = []
        for item in data_list:
            if item == anchor:
                continue
            if item['box'][0][1] > anchor_y_bottom:
                item_x_center = (item['box'][0][0] + item['box'][1][0]) / 2
                if abs(item_x_center - anchor_x_center) < x_var:
                    candidates.append(item)

        candidates.sort(key=lambda x: x['box'][0][1])
        result_lines = [c['text'] for c in candidates[:lookahead]]
        return result_lines


class BaseSIMStrategy:
    def cleanup_common(self, data):
        if data.get('Nama'):
            data['Nama'] = re.sub(r'^\d+[\.,\s]+', '', data['Nama']).strip()
            data['Nama'] = re.sub(r'[^A-Z\s\.\']', '', data['Nama'].upper())

        if data.get('Jenis Kelamin'):
            val = data['Jenis Kelamin'].upper()
            if 'PRIA' in val or 'LAKI' in val:
                data['Jenis Kelamin'] = 'LAKI-LAKI'
            elif 'WANITA' in val or 'PEREMPUAN' in val:
                data['Jenis Kelamin'] = 'PEREMPUAN'
        return data

    def is_garbage(self, text):
        garbage = [
            "PASSENGER", "PERSONAL", "GOODS", "DRIVING", "LICENSE",
            "SURAT", "IZIN", "MENGEMUDI", "MOBIL", "PENUMPANG",
            "MOTOR", "ANGONNA", "NAMA"
        ]
        return any(g in text.upper() for g in garbage)


class LegacySIMStrategy(BaseSIMStrategy):
    def extract(self, texts, all_data_with_boxes):
        extracted_data = {}

        i = 0
        merged_texts = []
        while i < len(texts):
            current_text = texts[i]
            if re.match(r'^\d\.$', current_text) and i + 1 < len(texts):
                merged_texts.append(f"{current_text} {texts[i + 1]}")
                i += 2
            else:
                merged_texts.append(current_text)
                i += 1
        texts = merged_texts

        for text in texts:
            sim_num_match = re.search(r'\d{4}-\d{4}-\d{6}', text)
            if sim_num_match:
                extracted_data['Nomor SIM'] = sim_num_match.group(0)

            expiry_match = re.search(r'\b(\d{2}-\d{2}-20\d{2})\b', text)
            if expiry_match:
                extracted_data['Berlaku Sampai'] = expiry_match.group(0)

        field_indices = {
            '1': -1, '2': -1, '3': -1, '4': -1, '5': -1, '6': -1
        }
        for i, text in enumerate(texts):
            for key in field_indices.keys():
                if text.startswith(f"{key}."):
                    field_indices[key] = i

        if field_indices['1'] != -1:
            val = texts[field_indices['1']]
            extracted_data['Nama'] = re.sub(r'^1\.\s*', '', val)

        if field_indices['2'] != -1:
            val = texts[field_indices['2']]
            extracted_data['Tempat & Tgl. Lahir'] = re.sub(r'^2\.\s*', '', val)

        if field_indices['3'] != -1:
            val = texts[field_indices['3']]
            extracted_data['Gol. Darah - Kelamin'] = re.sub(r'^3\.\s*', '', val)

        if field_indices['6'] != -1:
            val = texts[field_indices['6']]
            extracted_data['Provinsi'] = re.sub(r'^6\.\s*', '', val)

        if field_indices['4'] != -1:
            start = field_indices['4']
            end = len(texts)

            if field_indices['5'] != -1:
                end = field_indices['5']
            elif field_indices['6'] != -1:
                end = field_indices['6']

            addr_block = texts[start:end]
            potential_job = addr_block[-1]
            regex_loc = r'KOTA|KAB|KEC|KEL|DS|RT|RW|,'

            if field_indices['5'] == -1 and not re.search(regex_loc, potential_job):
                extracted_data['Pekerjaan'] = potential_job
                addr_block = addr_block[:-1]
            elif field_indices['5'] != -1:
                val = texts[field_indices['5']]
                extracted_data['Pekerjaan'] = re.sub(r'^5\.\s*', '', val)

            raw_addr = [re.sub(r'^4\.\s*', '', addr_block[0])] + addr_block[1:]
            extracted_data['raw_address_lines'] = raw_addr

        return extracted_data


class SmartSIMStrategy(BaseSIMStrategy):
    def extract(self, texts, all_data_with_boxes):
        extracted_data = {}

        sim_num = None
        for item in all_data_with_boxes:
            clean_text = item['text'].replace(" ", "")
            if re.match(r'^\d{12,16}$', clean_text):
                sim_num = clean_text
                break
        if sim_num:
            extracted_data['Nomor SIM'] = sim_num

        for i, item in enumerate(all_data_with_boxes):
            text = item['text']
            dob_match = re.search(r'([A-Z\s]+),\s*(\d{2}-\d{2}-\d{4})', text)

            if dob_match and not self.is_garbage(text):
                extracted_data['Tempat Lahir'] = dob_match.group(1).strip()
                extracted_data['Tanggal Lahir'] = dob_match.group(2).strip()

                if i > 0 and 'Nama' not in extracted_data:
                    for back_step in range(1, 4):
                        if i - back_step < 0:
                            break
                        prev = all_data_with_boxes[i - back_step]
                        prev_txt = prev['text']
                        if (len(prev_txt) > 2 and
                                not self.is_garbage(prev_txt) and
                                "TEMPAT" not in prev_txt.upper()):
                            extracted_data['Nama'] = prev_txt
                            break
                break

        if 'Nama' not in extracted_data:
            name_lines = GeometryUtils.find_value_below(
                "Nama", all_data_with_boxes, lookahead=1
            )
            if name_lines and not self.is_garbage(name_lines[0]):
                extracted_data['Nama'] = name_lines[0]

        sex_search = " ".join(texts).upper()
        if "PRIA" in sex_search:
            extracted_data['Jenis Kelamin'] = "LAKI-LAKI"
        elif "WANITA" in sex_search:
            extracted_data['Jenis Kelamin'] = "PEREMPUAN"

        blood_match = re.search(r'GOL\.?\s*DARAH\s*([A-Z0-9]+)', sex_search)
        if blood_match:
            extracted_data['Gol. Darah'] = blood_match.group(1)

        addr_lines = GeometryUtils.find_value_below(
            "Alamat", all_data_with_boxes, lookahead=6, x_var=600
        )
        if addr_lines:
            clean_addr = []
            for line in addr_lines:
                if "Pekerjaan" in line or "Provinsi" in line:
                    break
                if self.is_garbage(line) or "Gol. Darah" in line:
                    continue
                clean_addr.append(line)
            extracted_data['raw_address_lines'] = clean_addr

        job_keywords = [
            "KARYAWAN", "WIRASWASTA", "PELAJAR", "MAHASISWA",
            "PNS", "PEGAWAI", "BURUH", "IBU RUMAH"
        ]
        job_val = GeometryUtils.find_value_to_right(
            "Pekerjaan", all_data_with_boxes, max_x=400
        )
        if not job_val:
            for text in texts:
                if any(k in text.upper() for k in job_keywords):
                    job_val = text
                    break
        extracted_data['Pekerjaan'] = job_val

        dob = extracted_data.get('Tanggal Lahir', '')
        for text in texts[::-1]:
            match = re.search(r'\d{2}-\d{2}-20\d{2}', text)
            if match:
                date_found = match.group(0)
                if date_found != dob:
                    extracted_data['Berlaku Sampai'] = date_found
                    break

        return extracted_data


class SIMExtractor:
    def __init__(self):
        self.legacy_strategy = LegacySIMStrategy()
        self.smart_strategy = SmartSIMStrategy()

    def detect_version(self, texts):
        full_text = " ".join(texts)
        if (re.search(r'1\.\s+[A-Z]', full_text) or
                re.search(r'2\.\s+[A-Z]', full_text)):
            return "LEGACY"
        return "SMART"

    def process_sim(self, ocr_result):
        if not ocr_result or not ocr_result[0]:
            return None

        data = ocr_result[0]
        boxes = data.get('dt_polys', [])
        texts = data.get('rec_texts', [])

        if not texts:
            return None

        all_data = []
        for i, (box, text) in enumerate(zip(boxes, texts)):
            all_data.append({
                'id': i,
                'box': np.array(box).astype(np.int32),
                'text': text,
                'y_center': (box[0][1] + box[2][1]) / 2
            })

        version = self.detect_version(texts)
        strategy = (
            self.legacy_strategy if version == "LEGACY"
            else self.smart_strategy
        )

        extracted_raw = strategy.extract(texts, all_data)
        final_data = self.post_process_common(extracted_raw)
        return strategy.cleanup_common(final_data)

    def _parse_address_block(self, address_lines, all_data):
        full_address = " ".join(address_lines).upper()
        address_dict = {
            "name": None, "rt_rw": None, "kel_desa": None,
            "kecamatan": None, "kabupaten": None
        }

        rt_rw_match = re.search(
            r"(?:RT|RW)\s*[\./]?\s*(\d+).*?(?:RT|RW)\s*[\./]?\s*(\d+)",
            full_address
        )
        if rt_rw_match:
            address_dict["rt_rw"] = (
                f"{rt_rw_match.group(1)}/{rt_rw_match.group(2)}"
            )
            full_address = full_address.replace(rt_rw_match.group(0), "")
        else:
            simple_slash = re.search(
                r"(?:RT)?\s*(\d{1,3})\s*[/-]\s*(\d{1,3})", full_address
            )
            if simple_slash:
                address_dict["rt_rw"] = (
                    f"{simple_slash.group(1)}/{simple_slash.group(2)}"
                )
                full_address = full_address.replace(simple_slash.group(0), "")

        keywords = {
            "kabupaten": r"(?:KOTA|KABUPATEN|KAB)\.?\s+([A-Z\s]+?)"
                         r"(?=\s+KEC|\s+KEL|\s+DS|$)",
            "kecamatan": r"(?:KEC|KECAMATAN)\.?\s+([A-Z\s]+?)"
                         r"(?=\s+KOTA|\s+KAB|\s+KEL|\s+DS|$)",
            "kel_desa": r"(?:KEL|DESA|DS)\.?\s+([A-Z\s]+?)"
                        r"(?=\s+KEC|\s+KAB|\s+KOTA|RT|$)"
        }

        for key, pattern in keywords.items():
            match = re.search(pattern, full_address)
            if match:
                address_dict[key] = match.group(1).strip()
                full_address = full_address.replace(match.group(0), "")

        if not address_dict['kabupaten']:
            words = full_address.split()
            if words and len(words[-1]) > 3:
                potential_city = words[-1]
                address_dict['kabupaten'] = potential_city

                if (address_dict['kecamatan'] and
                        address_dict['kecamatan'].endswith(potential_city)):
                    new_kec = address_dict['kecamatan'][:-len(potential_city)]
                    address_dict['kecamatan'] = new_kec.strip()

                full_address = re.sub(
                    re.escape(potential_city) + r"$", "", full_address
                )

        if not address_dict['kecamatan']:
            comma_match = re.search(r",\s*([A-Z\s\.]+)$", full_address)
            if comma_match:
                address_dict['kecamatan'] = comma_match.group(1).strip()
                full_address = re.sub(
                    re.escape(comma_match.group(0)) + r"$", "", full_address
                )

        address_dict["name"] = re.sub(r'\s+', ' ', full_address).strip(" ,.")
        return address_dict

    def post_process_common(self, extracted_data):
        if 'Tempat & Tgl. Lahir' in extracted_data:
            parts = extracted_data['Tempat & Tgl. Lahir'].split(',', 1)
            extracted_data['Tempat Lahir'] = parts[0].strip()
            if len(parts) > 1:
                extracted_data['Tanggal Lahir'] = parts[1].strip()
            del extracted_data['Tempat & Tgl. Lahir']

        if 'Gol. Darah - Kelamin' in extracted_data:
            raw = extracted_data['Gol. Darah - Kelamin']
            if 'PRIA' in raw:
                extracted_data['Jenis Kelamin'] = 'LAKI-LAKI'
            elif 'WANITA' in raw:
                extracted_data['Jenis Kelamin'] = 'PEREMPUAN'

            clean_raw = raw.replace('PRIA', '').replace('WANITA', '').replace('-', '').strip()
            if len(clean_raw) <= 2:
                extracted_data['Gol. Darah'] = clean_raw
            del extracted_data['Gol. Darah - Kelamin']

        if 'raw_address_lines' in extracted_data:
            parsed = self._parse_address_block(
                extracted_data['raw_address_lines'], extracted_data
            )
            extracted_data.update(parsed)
            del extracted_data['raw_address_lines']
        else:
            extracted_data.update({
                "name": None, "rt_rw": None, "kel_desa": None,
                "kecamatan": None, "kabupaten": None
            })

        return extracted_data


def format_sim_to_json(data):
    if not data:
        return {
            "status": 400, "error": True,
            "message": "Failed to extract SIM data"
        }

    return {
        "status": 200,
        "error": False,
        "message": "SIM OCR Processed Successfully",
        "data": {
            "document_type": "SIM",
            "nomor": data.get("Nomor SIM"),
            "nama": data.get("Nama"),
            "tempat_lahir": data.get("Tempat Lahir"),
            "tgl_lahir": data.get("Tanggal Lahir"),
            "jenis_kelamin": data.get("Jenis Kelamin"),
            "agama": None,
            "status_perkawinan": None,
            "pekerjaan": data.get("Pekerjaan"),
            "kewarganegaraan": None,
            "alamat": {
                "name": data.get("name"),
                "rt_rw": data.get("rt_rw"),
                "kel_desa": data.get("kel_desa"),
                "kecamatan": data.get("kecamatan"),
                "kabupaten": data.get("kabupaten"),
                "provinsi": data.get("Provinsi")
            }
        }
    }