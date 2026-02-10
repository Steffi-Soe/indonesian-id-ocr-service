import re
import difflib
import numpy as np
from typing import List, Dict, Any, Optional

class GeometryUtils:
    @staticmethod
    def cluster_into_rows(
        data_list: List[Dict],
        y_threshold: int = 20
    ) -> List[List[Dict]]:
        if not data_list:
            return []

        sorted_data = sorted(data_list, key=lambda x: x['y_center'])
        rows = []

        current_row = [sorted_data[0]]

        for item in sorted_data[1:]:
            avg_y = sum(i['y_center'] for i in current_row) / len(current_row)

            if abs(item['y_center'] - avg_y) < y_threshold:
                current_row.append(item)
            else:
                current_row.sort(key=lambda x: x['box'][0][0])
                rows.append(current_row)
                current_row = [item]

        if current_row:
            current_row.sort(key=lambda x: x['box'][0][0])
            rows.append(current_row)

        return rows


class FuzzyMatcher:
    ANCHORS = {
        'NAMA': ['Nama', 'Name', 'NamaName'],
        'TTL': ['Tempat', 'Tgl', 'Lahir', 'Birth', 'Place', 'Date'],
        'GOL_DARAH': ['Gol', 'Darah', 'Blood', 'Type', 'Daah'],
        'JK': ['Jenis', 'Kelamin', 'Sex', 'Ketamin', 'Gender'],
        'ALAMAT': ['Alamat', 'Address', 'Alamrrat', 'Jalan', 'Jl.'],
        'PEKERJAAN': [
            'Pekerjaan', 'Occupation', 'eerjaan', 'Kerja', 'Wiraswasta',
            'Karyawan', 'Pelajar', 'Mahasiswa', 'PNS', 'Buruh', 'Mengurus'
        ],
        'PENERBIT': [
            'Diterbitkan', 'Issued', 'Oleh', 'Dierbtkan',
            'SATPAS', 'POLRES', 'POLDA'
        ],
    }

    JOB_KEYWORDS = [
        'WIRASWASTA', 'PELAJAR', 'MAHASISWA', 'KARYAWAN', 'BURUH',
        'PEGAWAI', 'PNS', 'POLRI', 'TNI', 'MENGURUS', 'DOKTER', 'BIDAN',
        'SWASTA', 'GURU', 'DOSEN', 'PEDAGANG', 'NELAYAN', 'PETANI'
    ]

    @staticmethod
    def identify_field(text: str, threshold: float = 0.55) -> Optional[str]:
        if not text:
            return None
        clean_text = re.sub(r'[^a-zA-Z]', '', text).lower()
        if len(clean_text) < 3:
            return None

        best_ratio = 0.0
        best_key = None

        for key, variants in FuzzyMatcher.ANCHORS.items():
            for var in variants:
                clean_var = re.sub(r'[^a-zA-Z]', '', var).lower()
                ratio = difflib.SequenceMatcher(None, clean_text, clean_var).ratio()
                if clean_var in clean_text:
                    ratio = max(ratio, 0.85)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_key = key

        if best_ratio >= threshold:
            return best_key
        return None

    @staticmethod
    def is_job(text: str) -> bool:
        text_upper = text.upper()
        if any(job in text_upper for job in FuzzyMatcher.JOB_KEYWORDS):
            return True
        if "KARY" in text_upper and "SWASTA" in text_upper:
            return True
        return False


class BaseSIMStrategy:
    GARBAGE_FRAGMENTS = [
        "SURAT", "IZIN", "MENGEMUDI", "DRIVING", "LICENSE",
        "INDONESIA", "POLRI", "KEPOLISIAN",
        "PASSENGER", "PERSONAL", "GOODS", "MOBIL", "PENUMPANG", "PRIBADI",
        "ANGONNA", "MOTOR", "VEHICLE", "PLACE", "DATE", "BIRTH", "BLOOD",
        "TYPE", "SAY", "DIERBTKAN", "ISSUED", "ANGKUTAN", "BARANG", "UMUM"
    ]

    def cleanup_common(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if data.get('Nama'):
            data['Nama'] = re.sub(r'^[\d\.\:\s]+', '', data['Nama']).strip()
            data['Nama'] = re.sub(r'[^A-Z\s\.\']', '', data['Nama'].upper())

        jk_raw = data.get('Jenis Kelamin', '') or data.get('Gol. Darah - Kelamin', '')
        if jk_raw:
            jk_upper = str(jk_raw).upper()
            if 'PRIA' in jk_upper or 'LAKI' in jk_upper:
                data['Jenis Kelamin'] = 'LAKI-LAKI'
            elif 'WANITA' in jk_upper or 'PEREMPUAN' in jk_upper:
                data['Jenis Kelamin'] = 'PEREMPUAN'

        if 'Gol. Darah - Kelamin' in data:
            del data['Gol. Darah - Kelamin']
        return data

    def is_garbage(self, text: str) -> bool:
        if not text: return True
        text_upper = text.upper()
        if len(text) < 2: return True
        for g in self.GARBAGE_FRAGMENTS:
            if g in text_upper:
                if any(x in text_upper for x in ["MOBIL", "PASSENGER", "PRIBADI", "GOODS", "DRIVING", "LICENSE", "SURAT IZIN"]):
                    return True
        return False


class LegacySIMStrategy(BaseSIMStrategy):
    def extract(self, texts: List[str], all_data_with_boxes: List[Dict]) -> Dict[str, Any]:
        extracted_data = {}
        rows = GeometryUtils.cluster_into_rows(all_data_with_boxes, y_threshold=20)
        current_section = 0
        address_accumulator = []

        for row in rows:
            row_text = " ".join([x['text'] for x in row]).strip()
            if not row_text: continue

            section_match = re.search(r'\b([1-6])\.', row_text)
            if section_match:
                try:
                    current_section = int(section_match.group(1))
                    clean_val = re.sub(rf'{current_section}\.\s*', '', row_text).strip()
                except ValueError:
                    clean_val = row_text
            else:
                if (current_section == 4 and FuzzyMatcher.is_job(row_text) and not self.is_garbage(row_text)):
                    current_section = 5
                    clean_val = row_text
                else:
                    clean_val = row_text

            if current_section == 1 and clean_val and len(clean_val) > 2:
                extracted_data['Nama'] = clean_val
            elif current_section == 2 and clean_val:
                extracted_data['Tempat & Tgl. Lahir'] = clean_val
            elif current_section == 3 and clean_val:
                match_jk = re.search(r'([ABO]+)\s*[-]*\s*(PRIA|WANITA|LAKI|PEREMPUAN)', clean_val.upper())
                if match_jk:
                    extracted_data['Gol. Darah'] = match_jk.group(1)
                    extracted_data['Jenis Kelamin'] = match_jk.group(2)
                else:
                    extracted_data['Gol. Darah - Kelamin'] = clean_val
            elif current_section == 4 and clean_val:
                if clean_val.replace('.', '').strip() == str(current_section): continue
                if not self.is_garbage(clean_val):
                    address_accumulator.append(clean_val)
            elif current_section == 5 and clean_val:
                if clean_val.replace('.', '').strip() == str(current_section): continue
                extracted_data['Pekerjaan'] = clean_val
            elif current_section == 6 and clean_val:
                extracted_data['Provinsi'] = clean_val

            sim_match = re.search(r'\d{4}-\d{4}-\d{6}', row_text)
            if sim_match: extracted_data['Nomor SIM'] = sim_match.group(0)
            expiry_match = re.search(r'\b(\d{2}-\d{2}-20\d{2})\b', row_text)
            if expiry_match:
                dob = extracted_data.get('Tempat & Tgl. Lahir', '')
                if expiry_match.group(0) not in dob:
                    extracted_data['Berlaku Sampai'] = expiry_match.group(0)

        if address_accumulator:
            extracted_data['raw_address_lines'] = address_accumulator
        return extracted_data


class SmartSIMStrategy(BaseSIMStrategy):
    def extract(self, texts: List[str], all_data_with_boxes: List[Dict]) -> Dict[str, Any]:
        extracted_data = {}
        rows = GeometryUtils.cluster_into_rows(all_data_with_boxes)
        row_texts = [" ".join([x['text'] for x in row]).strip() for row in rows]

        for t in row_texts:
            clean = t.replace(" ", "").replace("-", "")
            if re.match(r'^\d{12,16}$', clean):
                extracted_data['Nomor SIM'] = clean
                break

        tagged_rows = []
        for idx, text in enumerate(row_texts):
            ftype = FuzzyMatcher.identify_field(text)
            tagged_rows.append({'type': ftype, 'text': text, 'index': idx})

        nama_idx = self._find_anchor_index(tagged_rows, 'NAMA')
        if nama_idx is not None:
            extracted_data['Nama'] = self._find_value_forward(
                tagged_rows, nama_idx + 1, 2, ['TTL', 'ALAMAT']
            )

        ttl_idx = self._find_anchor_index(tagged_rows, 'TTL')
        if ttl_idx is not None:
            ttl_raw = self._find_value_forward(
                tagged_rows, ttl_idx + 1, 5, ['GOL_DARAH', 'JK', 'ALAMAT']
            )
            if ttl_raw: self._parse_ttl(ttl_raw, extracted_data)

        gd_idx = self._find_anchor_index(tagged_rows, 'GOL_DARAH')
        jk_idx = self._find_anchor_index(tagged_rows, 'JK')
        search_start = max(gd_idx or -1, jk_idx or -1) + 1
        
        if search_start > 0:
            limit = min(search_start + 4, len(row_texts))
            for i in range(search_start, limit):
                row = row_texts[i]
                if self.is_garbage(row): continue
                if FuzzyMatcher.identify_field(row) == 'ALAMAT': break
                
                clean_row = row.replace("-", "").strip().upper()
                if clean_row in ['A', 'B', 'AB', 'O'] and 'Gol. Darah' not in extracted_data:
                    extracted_data['Gol. Darah'] = clean_row
                    continue
                if 'PRIA' in row.upper() or 'LAKI' in row.upper():
                    extracted_data['Jenis Kelamin'] = 'LAKI-LAKI'
                elif 'WANITA' in row.upper() or 'PEREMPUAN' in row.upper():
                    extracted_data['Jenis Kelamin'] = 'PEREMPUAN'

        pekerjaan_idx = self._find_anchor_index(tagged_rows, 'PEKERJAAN')
        if pekerjaan_idx is not None:
            extracted_data['Pekerjaan'] = self._find_value_forward(
                tagged_rows, pekerjaan_idx + 1, 3, ['PENERBIT']
            )

        alamat_idx = self._find_anchor_index(tagged_rows, 'ALAMAT')
        if alamat_idx is not None:
            start = alamat_idx + 1
            stop_idx = pekerjaan_idx if pekerjaan_idx else len(row_texts)
            
            if stop_idx == len(row_texts):
                 for k in range(start, len(row_texts)):
                     if FuzzyMatcher.is_job(row_texts[k]):
                         stop_idx = k
                         break
            
            addr_lines = []
            for i in range(start, stop_idx):
                row = row_texts[i]
                if FuzzyMatcher.identify_field(row) in ['PEKERJAAN', 'PENERBIT']: break
                if "SATPAS" in row.upper() or "POLRES" in row.upper() or "POLDA" in row.upper(): continue
                if not self.is_garbage(row):
                    addr_lines.append(row)
            extracted_data['raw_address_lines'] = addr_lines

        if 'Berlaku Sampai' not in extracted_data:
            full_blob = " ".join(texts)
            dates = re.findall(r'(\d{2})[\s\.-]*(\d{2})[\s\.-]*(20\d{2})', full_blob)
            valid_expiry = None
            if dates:
                for d, m, y in dates:
                    try:
                        if int(y) > 2018: valid_expiry = f"{d}-{m}-{y}"
                    except ValueError: continue
            if valid_expiry: extracted_data['Berlaku Sampai'] = valid_expiry

        return extracted_data

    def _find_anchor_index(self, tagged_rows, atype):
        for row in tagged_rows:
            if row['type'] == atype: return row['index']
        return None

    def _find_value_forward(self, tagged_rows, start_idx, max_lookahead, stop_types=None):
        limit = min(start_idx + max_lookahead, len(tagged_rows))
        for i in range(start_idx, limit):
            row = tagged_rows[i]
            if stop_types and row['type'] in stop_types: return None
            if self.is_garbage(row['text']): continue
            if len(row['text']) < 3 and not re.search(r'\d', row['text']): continue
            return row['text']
        return None

    def _parse_ttl(self, text, data):
        if not text: return
        text = text.strip()
        if ',' in text:
            parts = text.split(',', 1)
            data['Tempat Lahir'] = parts[0].strip()
            if len(parts) > 1: data['Tanggal Lahir'] = parts[1].strip()
            return
        date_match = re.search(r'(\d{1,2})[-\s]?(\d{1,2})[-\s]?(19\d{2}|20\d{2})$', text)
        if date_match:
            d, m, y = date_match.groups()
            data['Tempat Lahir'] = text[:date_match.start()].strip()
            data['Tanggal Lahir'] = f"{d}-{m}-{y}"
        else:
            data['Tempat Lahir'] = text


class SIMExtractor:
    def __init__(self):
        self.legacy_strategy = LegacySIMStrategy()
        self.smart_strategy = SmartSIMStrategy()
        self.cities = {
            'JAKARTA', 'BOGOR', 'DEPOK', 'TANGERANG', 'BEKASI', 'BANDUNG',
            'SEMARANG', 'SURABAYA', 'MEDAN', 'MAKASSAR', 'BALIKPAPAN',
            'DENPASAR', 'SLEMAN', 'BANTUL', 'KULON PROGO', 'SERANG',
            'CILEGON', 'CIMAHI', 'SUKABUMI', 'BATAM', 'KUPANG', 'PONOROGO',
            'MALANG', 'SOLO', 'SURAKARTA', 'YOGYAKARTA', 'PALEMBANG',
            'PEKANBARU', 'PADANG', 'LAMPUNG', 'JAMBI', 'BENGKULU', 'ACEH',
            'MATARAM', 'JAYAPURA', 'MANADO', 'AMBON', 'KENDARI', 'PALU'
        }

    def detect_version(self, texts: List[str]) -> str:
        full_text = " ".join(texts)
        if re.search(r'1\.\s', full_text) or re.search(r'2\.\s', full_text):
            return "LEGACY"
        return "SMART"

    def process_sim(self, ocr_result: List[Dict]) -> Optional[Dict[str, Any]]:
        if not ocr_result or not isinstance(ocr_result, list): return None
        data = ocr_result[0]
        if not isinstance(data, dict): return None
        boxes = data.get('dt_polys', [])
        texts = data.get('rec_texts', [])
        if not texts or not boxes or len(boxes) != len(texts): return None

        all_data = []
        for i, (box, text) in enumerate(zip(boxes, texts)):
            try:
                np_box = np.array(box, dtype=np.int32)
                if np_box.shape != (4, 2): continue
                all_data.append({
                    'id': i, 'box': np_box, 'text': str(text),
                    'y_center': (np_box[0][1] + np_box[2][1]) / 2
                })
            except Exception: continue
        
        if not all_data: return None
        version = self.detect_version(texts)
        strategy = self.legacy_strategy if version == "LEGACY" else self.smart_strategy

        try:
            extracted_raw = strategy.extract(texts, all_data)
            final_data = self.post_process_common(extracted_raw)
            return strategy.cleanup_common(final_data)
        except Exception: return None

    def _parse_address_block(self, address_lines: List[str]) -> Dict[str, Optional[str]]:
        addr = {
            "name": None, "rt_rw": None, "kel_desa": None,
            "kecamatan": None, "kabupaten": None, "provinsi": None
        }
        if not address_lines:
            return addr

        clean_lines = []
        for line in address_lines:
            line = re.sub(r'^(Alamat|Address)[\s\:\.]*', '', line, flags=re.IGNORECASE).strip()
            line = re.sub(r'^[4]\.\s*', '', line).strip()
            if not line: continue
            if FuzzyMatcher.is_job(line): continue
            clean_lines.append(line)

        if not clean_lines:
            return addr

        city_index = len(clean_lines) 
        for idx in range(len(clean_lines) - 1, -1, -1):
            line_u = clean_lines[idx].upper()
            is_city = any(c in line_u for c in self.cities)
            if 'KOTA' in line_u or 'KAB' in line_u or 'JAKARTA' in line_u: is_city = True
            
            if is_city:
                if not addr['kabupaten']:
                    addr['kabupaten'] = clean_lines[idx]
                else:
                    pass
                city_index = idx

        street_parts = []
        state = 0
        
        rt_pivot_re = re.compile(r'(?:RT|RW|R\.T|R\.W)[\s\.\:]*(\d{1,4})', re.IGNORECASE)
        rt_sep_re = re.compile(r'^[\s\/\-\|lI1]+(\d{1,4})', re.IGNORECASE)
        rw_residue_re = re.compile(r'^\s*(?:RW|RW\.|W\.|RW:)[\s\.\:]*(\d{1,4})', re.IGNORECASE)

        street_prefixes = ('JL', 'JALAN', 'GG', 'GANG', 'KP', 'KMP', 'KOMP', 'DUSUN', 'DSN', 'BLK', 'BLOK', 'NO')

        for idx, line in enumerate(clean_lines):
            if idx >= city_index: break
            
            line_u = line.upper()
            
            if 'KEC' in line_u and 'KECIL' not in line_u:
                val = re.sub(r'\b(KEC|KECAMATAN)\b\.?', '', line, flags=re.IGNORECASE).strip()
                addr['kecamatan'] = val
                state = 1
                continue

            is_kel_prefix = False
            for p in ['KEL', 'DESA', 'DS']:
                 if re.match(rf'^{p}\b', line_u) or re.match(rf'^{p}\.', line_u):
                     is_kel_prefix = True
                     break
            
            rt_match = rt_pivot_re.search(line)
            
            if rt_match:
                state = 1
                start, end = rt_match.span()
                prefix = line[:start].strip()
                match_val = rt_match.group(1)
                residue = line[end:]
                
                sep_match = rt_sep_re.match(residue)
                rw_val = None
                
                if sep_match:
                    rw_val = sep_match.group(1)
                    residue = residue[sep_match.end():]
                else:
                    rw_match = rw_residue_re.search(residue)
                    if rw_match:
                        rw_val = rw_match.group(1)
                        residue = residue[rw_match.end():]
                
                if rw_val:
                    addr['rt_rw'] = f"{match_val}/{rw_val}"
                else:
                    addr['rt_rw'] = match_val
                
                if is_kel_prefix:
                    cleaned_prefix = re.sub(r'\b(KEL|DESA|DS)\b\.?', '', prefix, flags=re.IGNORECASE).strip()
                    addr['kel_desa'] = cleaned_prefix
                elif prefix:
                    street_parts.append(prefix)
                
                residue = residue.strip()
                if len(residue) > 2:
                    residue = re.sub(r'^[\-\,\.]+', '', residue).strip()
                    if not addr['kel_desa']:
                        addr['kel_desa'] = residue
                    elif not addr['kecamatan']:
                        addr['kecamatan'] = residue
                continue

            
            if is_kel_prefix:
                val = re.sub(r'\b(KEL|DESA|DS)\b\.?', '', line, flags=re.IGNORECASE).strip()
                addr['kel_desa'] = val
                state = 1
                continue
            
            if state == 0:
                starts_with_street = any(line_u.startswith(p) for p in street_prefixes)
                
                if ',' in line and not starts_with_street:
                    parts = line.split(',', 1)
                    p1 = parts[0].strip()
                    p2 = parts[1].strip()
                    if not addr['kel_desa']: addr['kel_desa'] = p1
                    if not addr['kecamatan']: addr['kecamatan'] = p2
                    state = 1
                else:
                    street_parts.append(line)
            else:
                if not addr['kel_desa']:
                    addr['kel_desa'] = line
                elif not addr['kecamatan']:
                    addr['kecamatan'] = line
                else:
                    addr['kecamatan'] += " " + line

        if street_parts:
            addr['name'] = " ".join(street_parts)

        return addr

    def post_process_common(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        if 'Tempat & Tgl. Lahir' in extracted_data:
            val = extracted_data['Tempat & Tgl. Lahir']
            if ',' in val:
                parts = val.split(',', 1)
                extracted_data['Tempat Lahir'] = parts[0].strip()
                if len(parts) > 1: extracted_data['Tanggal Lahir'] = parts[1].strip()
            else:
                extracted_data['Tempat Lahir'] = val
            del extracted_data['Tempat & Tgl. Lahir']

        if 'raw_address_lines' in extracted_data:
            parsed = self._parse_address_block(extracted_data['raw_address_lines'])
            extracted_data['alamat'] = parsed
            del extracted_data['raw_address_lines']
        else:
            extracted_data['alamat'] = {
                "name": None, "rt_rw": None, "kel_desa": None,
                "kecamatan": None, "kabupaten": None, "provinsi": None
            }
            if 'Provinsi' in extracted_data:
                extracted_data['alamat']['provinsi'] = extracted_data['Provinsi']
        return extracted_data


def format_sim_to_json(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not data:
        return {"status": 400, "error": True, "message": "Failed to extract SIM data"}

    addr = data.get('alamat', {})
    if not isinstance(addr, dict): addr = {}

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
                "name": addr.get("name"),
                "rt_rw": addr.get("rt_rw"),
                "kel_desa": addr.get("kel_desa"),
                "kecamatan": addr.get("kecamatan"),
                "kabupaten": addr.get("kabupaten"),
                "provinsi": addr.get("provinsi")
            },
        }
    }