import re
import numpy as np

class SIMExtractor:
    def process_sim(self, ocr_result):
        if not ocr_result or not ocr_result[0]:
            return None
        
        result_dict = ocr_result[0]
        boxes = result_dict.get('dt_polys', [])
        texts = result_dict.get('rec_texts', [])

        if not texts:
            return None

        recognized_data = []
        for i, (box, text) in enumerate(zip(boxes, texts)):
            recognized_data.append({
                'id': i, 'box': np.array(box).astype(np.int32), 'text': text,
                'y_center': (box[0][1] + box[2][1]) / 2
            })
            
        structured_data = self.post_process(recognized_data)
        return self.cleanup_data(structured_data)

    def _parse_address_block(self, address_lines, all_data):
        full_address = " ".join(address_lines).upper()
        address_dict = {"name": None, "rt_rw": None, "kel_desa": None, "kecamatan": None, "kabupaten": None}

        rt_rw_match = re.search(r"RT\s*(\d+)\s*RW\s*(\d+)", full_address)
        if rt_rw_match:
            rt, rw = rt_rw_match.group(1).zfill(3), rt_rw_match.group(2).zfill(3)
            address_dict["rt_rw"] = f"{rt}/{rw}"
            full_address = full_address.replace(rt_rw_match.group(0), "")
        else:
            rt_rw_match = re.search(r"(?:RT)?\s*(\d{2,3}\s*/\s*\d{2,3})", full_address)
            if rt_rw_match:
                address_dict["rt_rw"] = re.sub(r'\s', '', rt_rw_match.group(1))
                full_address = full_address.replace(rt_rw_match.group(0), "").strip()

        keywords = {
            "kabupaten": r"(?:KOTA|KABUPATEN|KAB)\s+([A-Z\s/]+?)(?=\s+KEC|\s+KEL|\s+DS|$)",
            "kecamatan": r"(?:KEC|KECAMATAN)\s+([A-Z\s/]+?)(?=\s+KOTA|\s+KAB|\s+KEL|\s+DS|$)",
            "kel_desa": r"(?:KEL|DESA|DS)\s+([A-Z\s/]+?)(?=\s+KEC|\s+KAB|\s+KOTA|RT|$)"
        }
        for key, pattern in keywords.items():
            match = re.search(pattern, full_address)
            if match:
                address_dict[key] = match.group(1).strip()
                full_address = full_address.replace(match.group(0), "").strip()
        
        if address_dict.get("kecamatan") and not address_dict.get("kabupaten"):
            parts = address_dict["kecamatan"].split()
            if len(parts) > 1:
                birthplace = all_data.get("Tempat Lahir", "").upper()
                if birthplace and parts[-1] == birthplace:
                    address_dict["kabupaten"] = parts[-1]
                    address_dict["kecamatan"] = " ".join(parts[:-1])
                else:
                    address_dict["kabupaten"] = parts[-1]
                    address_dict["kecamatan"] = " ".join(parts[:-1])

        comma_match = re.search(r'([A-Z\s]+),\s+([A-Z]+)', full_address)
        if comma_match and not address_dict.get('kecamatan'):
            if len(comma_match.group(2)) > 3:
                if address_dict.get('kel_desa') is None:
                    address_dict['kel_desa'] = comma_match.group(1).strip()
                address_dict['kecamatan'] = comma_match.group(2).strip()
                full_address = full_address.replace(comma_match.group(0), '')
        
        address_dict["name"] = re.sub(r'\s+', ' ', full_address).strip(" ,.")
        if not address_dict["name"]:
            address_dict["name"] = None
            
        return address_dict

    def post_process(self, recognized_data):
        extracted_data = {}
        original_texts = [item['text'].strip() for item in recognized_data]

        texts = []
        i = 0
        while i < len(original_texts):
            current_text = original_texts[i]
            if re.match(r'^\d\.$', current_text) and i + 1 < len(original_texts):
                texts.append(f"{current_text} {original_texts[i+1]}")
                i += 2
            else:
                texts.append(current_text)
                i += 1

        for text in texts:
            sim_num_match = re.search(r'\d{4}-\d{4}-\d{6}', text)
            if sim_num_match:
                extracted_data['Nomor SIM'] = sim_num_match.group(0)
            
            expiry_date_match = re.search(r'\b(\d{2}-\d{2}-20\d{2})\b', text)
            if expiry_date_match:
                extracted_data['Berlaku Sampai'] = expiry_date_match.group(0)
        
        field_indices = {'1': -1, '2': -1, '3': -1, '4': -1, '5': -1, '6': -1}
        for i, text in enumerate(texts):
            for key in field_indices.keys():
                if text.startswith(f"{key}."):
                    field_indices[key] = i

        if field_indices['1'] != -1:
            extracted_data['Nama'] = re.sub(r'^1\.\s*', '', texts[field_indices['1']])
        if field_indices['2'] != -1:
            extracted_data['Tempat & Tgl. Lahir'] = re.sub(r'^2\.\s*', '', texts[field_indices['2']])
        if field_indices['3'] != -1:
            extracted_data['Gol. Darah - Kelamin'] = re.sub(r'^3\.\s*', '', texts[field_indices['3']])
        if field_indices['6'] != -1:
            extracted_data['Provinsi'] = re.sub(r'^6\.\s*', '', texts[field_indices['6']])

        temp_cleaned_data = self.cleanup_data(extracted_data.copy())

        if field_indices['4'] != -1:
            address_start = field_indices['4']
            
            address_end = len(texts)
            if field_indices['5'] != -1:
                address_end = field_indices['5']
            elif field_indices['6'] != -1:
                address_end = field_indices['6']

            address_text_block = texts[address_start:address_end]
            
            potential_job = address_text_block[-1]
            if field_indices['5'] == -1 and not re.search(r'KOTA|KAB|KEC|KEL|DS|RT|RW|,', potential_job):
                 extracted_data['Pekerjaan'] = potential_job
                 address_text_block = address_text_block[:-1]
            elif field_indices['5'] != -1:
                 extracted_data['Pekerjaan'] = re.sub(r'^5\.\s*', '', texts[field_indices['5']])

            address_lines = [re.sub(r'^4\.\s*', '', address_text_block[0])] + address_text_block[1:]
            parsed_address = self._parse_address_block(address_lines, temp_cleaned_data)
            extracted_data.update(parsed_address)
            
        return extracted_data

    def cleanup_data(self, data):
        cleaned = data.copy()
        if data.get('Tempat & Tgl. Lahir'):
            parts = data['Tempat & Tgl. Lahir'].split(',', 1)
            cleaned['Tempat Lahir'] = parts[0].strip()
            if len(parts) > 1:
                cleaned['Tanggal Lahir'] = parts[1].strip()
            if 'Tempat & Tgl. Lahir' in cleaned: del cleaned['Tempat & Tgl. Lahir']
        
        if data.get('Gol. Darah - Kelamin'):
            raw_field = data['Gol. Darah - Kelamin']
            parts = [p.strip() for p in raw_field.split('-') if p and p.strip()]
            
            gender = None
            if parts:
                last_part = parts[-1].upper()
                if last_part in ['PRIA', 'WANITA']:
                    gender = last_part
                    parts.pop()

                cleaned['Gol. Darah'] = parts[0] if parts else None
            
            if gender == 'PRIA':
                cleaned['Jenis Kelamin'] = 'LAKI-LAKI'
            elif gender == 'WANITA':
                cleaned['Jenis Kelamin'] = 'PEREMPUAN'
            else:
                cleaned['Jenis Kelamin'] = gender
                
            if 'Gol. Darah - Kelamin' in cleaned: del cleaned['Gol. Darah - Kelamin']
            
        return cleaned

def format_sim_to_json(data):
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