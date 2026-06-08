import re
import difflib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from thefuzz import process, fuzz
from date_normalizer import normalize_date_robust  # authoritative DD-MM-YYYY normalizer


# ---------------------------------------------------------------------------
# Canonical normalization maps (shared with ktp_extractor)
# ---------------------------------------------------------------------------

PEKERJAAN_CANONICAL: Dict[str, List[str]] = {
    "WIRASWASTA":           ["WIRASWASTA", "WIRAUSAHA", "WIRASWAST"],
    "PELAJAR/MAHASISWA":    ["PELAJAR", "MAHASISWA", "PELAJAR/MAHASISWA",
                             "PELAJARMAHASISWA"],
    "KARYAWAN SWASTA":      ["KARYAWAN SWASTA", "KARYAWAN", "KARY. SWASTA",
                             "KARY SWASTA", "KARYAWANSWASTA"],
    "PNS":                  ["PNS", "PEGAWAI NEGERI SIPIL", "PEGAWAI NEGERI", "P.N.S"],
    "TNI":                  ["TNI", "TENTARA NASIONAL INDONESIA", "TENTARA"],
    "POLRI":                ["POLRI", "POLISI"],
    "BURUH HARIAN LEPAS":   ["BURUH HARIAN LEPAS", "BURUH HARIAN", "BURUH LEPAS",
                             "CURLH HARIAN LEPAS", "CURLH HARIAN",
                             "CURUH HARIAN LEPAS", "DURUH HARIAN LEPAS"],
    "BURUH":                ["BURUH", "KULI"],
    "PEDAGANG":             ["PEDAGANG", "PENJUAL"],
    "PETANI":               ["PETANI"],
    "NELAYAN":              ["NELAYAN"],
    "GURU":                 ["GURU", "PENGAJAR"],
    "DOKTER":               ["DOKTER"],
    "BIDAN":                ["BIDAN"],
    "PERAWAT":              ["PERAWAT"],
    "DOSEN":                ["DOSEN"],
    "TIDAK BEKERJA":        ["TIDAK BEKERJA", "BELUM BEKERJA", "PENGANGGURAN"],
    "IBU RUMAH TANGGA":     ["IRT", "IBU RUMAH TANGGA", "IRUMAHTANGGA",
                             "MENGURUS RUMAH TANGGA", "MENGURUS RT", "RUMAH TANGGA"],
    "SUPIR":                ["SUPIR", "SOPIR", "DRIVER"],
    "OJEK":                 ["OJEK", "PENGEMUDI OJEK"],
    "SWASTA":               ["SWASTA"],
    "PEGAWAI SWASTA":       ["PEGAWAI SWASTA"],
}

# Flat list used for fast substring + fuzzy checks
_ALL_PEKERJAAN_TERMS: List[str] = sorted(
    {term.upper() for terms in PEKERJAAN_CANONICAL.values() for term in terms},
    key=len, reverse=True,   # longer terms first → prefer specific matches
)

# ---------------------------------------------------------------------------
# Indonesian regions (cities, regencies, provinces) for fuzzy city detection
# ---------------------------------------------------------------------------

INDONESIAN_REGIONS: List[str] = [
    # DKI Jakarta
    "JAKARTA", "JAKARTA BARAT", "JAKARTA TIMUR", "JAKARTA SELATAN",
    "JAKARTA UTARA", "JAKARTA PUSAT", "DKI JAKARTA",
    # Jawa Barat
    "BANDUNG", "BOGOR", "BEKASI", "DEPOK", "CIMAHI", "SUKABUMI",
    "CIREBON", "GARUT", "TASIKMALAYA", "CIAMIS", "KUNINGAN",
    "CIANJUR", "SUMEDANG", "MAJALENGKA", "SUBANG", "PURWAKARTA",
    "KARAWANG", "INDRAMAYU", "JAWA BARAT",
    # Banten
    "TANGERANG", "TANGERANG SELATAN", "SERANG", "CILEGON",
    "LEBAK", "PANDEGLANG", "BANTEN",
    # Jawa Tengah
    "SEMARANG", "SOLO", "SURAKARTA", "MAGELANG", "SALATIGA",
    "TEGAL", "PEKALONGAN", "KUDUS", "JEPARA", "DEMAK", "BLORA",
    "REMBANG", "PATI", "BOYOLALI", "KLATEN", "PURWOREJO",
    "KEBUMEN", "BANYUMAS", "CILACAP", "JAWA TENGAH",
    # DIY
    "YOGYAKARTA", "SLEMAN", "BANTUL", "GUNUNG KIDUL", "KULON PROGO",
    "DAERAH ISTIMEWA YOGYAKARTA",
    # Jawa Timur
    "SURABAYA", "MALANG", "SIDOARJO", "GRESIK", "MOJOKERTO",
    "PASURUAN", "PROBOLINGGO", "BANYUWANGI", "JEMBER", "KEDIRI",
    "BLITAR", "MADIUN", "PONOROGO", "NGAWI", "JOMBANG",
    "LAMONGAN", "BOJONEGORO", "TUBAN", "JAWA TIMUR",
    # Sumatera
    "MEDAN", "PADANG", "PEKANBARU", "PALEMBANG", "LAMPUNG",
    "BANDAR LAMPUNG", "BENGKULU", "JAMBI", "BATAM", "TANJUNGPINANG",
    "BANDA ACEH", "ACEH", "SUMATERA UTARA", "SUMATERA BARAT",
    "SUMATERA SELATAN", "RIAU", "KEPULAUAN RIAU", "BANGKA BELITUNG",
    # Kalimantan
    "BANJARMASIN", "BALIKPAPAN", "SAMARINDA", "PONTIANAK",
    "PALANGKARAYA", "TARAKAN", "KALIMANTAN BARAT",
    "KALIMANTAN SELATAN", "KALIMANTAN TIMUR", "KALIMANTAN TENGAH",
    "KALIMANTAN UTARA",
    # Sulawesi
    "MAKASSAR", "MANADO", "KENDARI", "PALU", "GORONTALO",
    "SULAWESI SELATAN", "SULAWESI UTARA", "SULAWESI TENGGARA",
    "SULAWESI TENGAH", "SULAWESI BARAT",
    # Bali & Nusa Tenggara
    "DENPASAR", "MATARAM", "KUPANG", "BALI",
    "NUSA TENGGARA BARAT", "NUSA TENGGARA TIMUR",
    # Maluku & Papua
    "AMBON", "JAYAPURA", "SORONG", "MANOKWARI",
    "MALUKU", "PAPUA", "PAPUA BARAT",
]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _split_rtrw_ocr(
    match_val: str,
    residue:   str,
) -> Optional[Tuple[str, str, str]]:
    """
    Recover a proper RT/RW pair when the '/' separator was OCR'd as '1'.

    E.g. ``match_val='0210'``, ``residue='6 JATIUWUNG'``
         → combined digit string '02106'
         → '1' at position 2 treated as '/'
         → returns ('02', '06', 'JATIUWUNG')

    Only called when the primary separator-regex failed to find an RW value,
    which prevents false splits on legitimate 3-digit RT values like '011'.

    Returns (rt_str, rw_str, new_residue) or None.
    """
    digits = re.sub(r'\D', '', match_val)
    residue_lead  = re.match(r'^\s*(\d+)', residue)
    residue_digits = residue_lead.group(1) if residue_lead else ""
    combined = digits + residue_digits

    # Need at least 5 digits: even the shortest RT/RW pair (1/1) with a
    # separator digit gives 3 digits total; a realistic '02/06' OCR'd
    # as '02106' yields 5.
    if len(combined) < 5:
        return None

    best: Optional[Tuple[str, str, str]] = None
    for i in range(1, len(combined) - 1):
        if combined[i] == '1':
            rt_c = combined[:i]
            rw_c = combined[i + 1:]
            if 1 <= len(rt_c) <= 3 and 1 <= len(rw_c) <= 3:
                new_residue = (
                    residue[residue_lead.end():].strip()
                    if residue_lead else residue
                )
                candidate = (rt_c, rw_c, new_residue)
                # Strongly prefer splits where both sides have ≥ 2 digits,
                # e.g. ('02','06') over ('0','206').  Return immediately on
                # a 2+2 match; keep iterating if only a 1+n match is found.
                if len(rt_c) >= 2 and len(rw_c) >= 2:
                    return candidate
                if best is None:
                    best = candidate

    return best


def normalize_pekerjaan(raw: str) -> str:
    """
    Normalize an OCR-extracted Pekerjaan value to its canonical form.

    Uses the PEKERJAAN_CANONICAL map with a two-pass strategy:
      1. Exact alias match (case-insensitive).
      2. ``fuzz.token_set_ratio`` fuzzy scan with threshold 72.
    Returns the original value if no match qualifies.
    """
    if not raw:
        return raw
    raw_u = raw.upper().strip()
    # Pass 1 – exact
    for canonical, aliases in PEKERJAAN_CANONICAL.items():
        if raw_u in [a.upper() for a in aliases]:
            return canonical
    # Pass 2 – fuzzy
    best_canonical, best_score = raw, 0
    for canonical, aliases in PEKERJAAN_CANONICAL.items():
        for alias in aliases:
            score = fuzz.token_set_ratio(raw_u, alias.upper())
            if score > best_score:
                best_score, best_canonical = score, canonical
    return best_canonical if best_score >= 72 else raw


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

class GeometryUtils:
    @staticmethod
    def cluster_into_rows(
        data_list: List[Dict],
        y_threshold: Optional[int] = None
    ) -> List[List[Dict]]:
        if not data_list:
            return []

        if y_threshold is None:
            heights  = [abs(item['box'][3][1] - item['box'][0][1]) for item in data_list]
            median_h = sorted(heights)[len(heights) // 2] if heights else 20
            y_threshold = max(10, int(median_h * 0.5))

        sorted_data = sorted(data_list, key=lambda x: x['y_center'])
        rows        = []
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


# ---------------------------------------------------------------------------
# Fuzzy field matcher
# ---------------------------------------------------------------------------

class FuzzyMatcher:
    ANCHORS = {
        'NAMA':      ['Nama', 'Name', 'NamaName'],
        'TTL':       ['Tempat', 'Lahir', 'Birth', 'Place', 'Date'],
        'GOL_DARAH': ['Darah', 'Blood', 'Type'],
        'JK':        ['Jenis', 'Kelamin', 'Sex', 'Gender'],
        'ALAMAT':    ['Alamat', 'Address', 'Alamrrat'],
        'PEKERJAAN': ['Pekerjaan', 'Occupation', 'eerjaan'],
        'PENERBIT':  [
            'Diterbitkan', 'Issued', 'Oleh', 'Dierbtkan',
            'SATPAS', 'POLRES', 'POLDA', 'KORLANTAS', 'METRO JAYA', 'METROJAYA',
        ],
    }

    @staticmethod
    def identify_field(text: str, threshold: float = 0.65) -> Optional[str]:
        if not text:
            return None
        clean_text = re.sub(r'[^a-zA-Z]', '', text).lower()
        if len(clean_text) < 4:
            return None

        best_ratio = 0.0
        best_key   = None

        for key, variants in FuzzyMatcher.ANCHORS.items():
            for var in variants:
                clean_var = re.sub(r'[^a-zA-Z]', '', var).lower()
                if len(clean_var) < 3:
                    continue
                ratio = difflib.SequenceMatcher(None, clean_text, clean_var).ratio()
                if clean_var in clean_text and len(clean_var) >= 4:
                    ratio = max(ratio, 0.90)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_key   = key

        return best_key if best_ratio >= threshold else None

    @staticmethod
    def is_job(text: str) -> bool:
        """
        Return True if *text* looks like a Pekerjaan (occupation) value.

        Strategy:
          1. Fast substring scan against all canonical aliases.
          2. Fuzzy fallback (``token_set_ratio`` ≥ 80) for OCR-noisy values.
        """
        if not text:
            return False
        text_upper = text.upper()

        # Pass 1 – exact / substring
        for term in _ALL_PEKERJAAN_TERMS:
            if term in text_upper:
                return True

        # Pass 2 – fuzzy (only for reasonably long strings to avoid noise)
        if len(text_upper) >= 4:
            _, score = process.extractOne(
                text_upper, _ALL_PEKERJAAN_TERMS, scorer=fuzz.token_set_ratio
            )
            return score >= 80

        return False


# ---------------------------------------------------------------------------
# Base strategy
# ---------------------------------------------------------------------------

class BaseSIMStrategy:
    def cleanup_common(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if data.get('Nama'):
            data['Nama'] = re.sub(r'^[\d\.\:\s]+', '', data['Nama']).strip()
            data['Nama'] = re.sub(r'[^A-Z\s\.\'\-]', '', data['Nama'].upper()).strip()
            if not data['Nama']:
                data['Nama'] = None

        jk_raw = data.get('Jenis Kelamin', '') or data.get('Gol. Darah - Kelamin', '')
        if jk_raw:
            jk_upper = str(jk_raw).upper()
            if 'PRIA' in jk_upper or 'LAKI' in jk_upper:
                data['Jenis Kelamin'] = 'LAKI-LAKI'
            elif 'WANITA' in jk_upper or 'PEREMPUAN' in jk_upper:
                data['Jenis Kelamin'] = 'PEREMPUAN'

        if 'Gol. Darah - Kelamin' in data:
            del data['Gol. Darah - Kelamin']

        # Normalize pekerjaan via canonical map
        if data.get('Pekerjaan'):
            data['Pekerjaan'] = normalize_pekerjaan(data['Pekerjaan'])

        # Clean kabupaten: remove short leading OCR artefacts
        addr = data.get('alamat')
        if isinstance(addr, dict) and addr.get('kabupaten'):
            addr['kabupaten'] = _clean_sim_kabupaten(addr['kabupaten'])

        return data

    def is_garbage(self, text: str) -> bool:
        if not text:
            return True
        text_upper = text.upper()
        if len(text) < 2:
            return True
        if "MOTOR"        in text_upper and "CC"            in text_upper: return True
        if "SEPEDA"       in text_upper and "MOTOR"         in text_upper: return True
        if "MOBIL"        in text_upper and "PENUMPANG"     in text_upper: return True
        if "PASSENGER"    in text_upper and "GOODS"         in text_upper: return True
        if "PLACE"        in text_upper and "BIRTH"         in text_upper: return True
        if "BLOOD"        in text_upper and "TYPE"          in text_upper: return True
        if any(x in text_upper for x in [
            "<= 250", "250 CC", "TRUK/BUS", "DRIVING LICENSE",
            "SURAT IZIN", "MENGEMUDI", "DITERBITKAN",
        ]):
            return True
        if text_upper.strip() in [
            "INDONESIA", "SURAT", "IZIN", "MENGEMUDI", "DRIVING", "LICENSE",
        ]:
            return True
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_sim_kabupaten(raw: str) -> str:
    """Strip leading short OCR tokens before known city/region keywords."""
    if not raw:
        return raw
    cleaned = re.sub(
        r'^[A-Z]{1,5}\s+(?=KOTA\b|KAB\b|KABUPATEN\b|JAKARTA\b|BANDUNG\b|SURABAYA\b)',
        '', raw.strip()
    )
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Legacy SIM strategy (numbered-section layout)
# ---------------------------------------------------------------------------

class LegacySIMStrategy(BaseSIMStrategy):
    def extract(self, texts: List[str], all_data_with_boxes: List[Dict]) -> Dict[str, Any]:
        extracted_data = {}
        rows      = GeometryUtils.cluster_into_rows(all_data_with_boxes)
        row_texts = [" ".join([x['text'] for x in row]).strip() for row in rows]

        current_section     = 0
        address_accumulator = []

        for idx, row_text in enumerate(row_texts):
            if not row_text:
                continue

            # Expiry date
            expiry_match = re.search(r'\b(\d{2}-\d{2}-20\d{2})\b', row_text)
            if expiry_match:
                dob = extracted_data.get('Tempat & Tgl. Lahir', '')
                if expiry_match.group(1) not in dob:
                    extracted_data['Berlaku Sampai'] = expiry_match.group(1)
                    row_text = row_text.replace(expiry_match.group(1), "").strip()

            if not row_text:
                continue

            # Penerbit
            if any(p in row_text.upper() for p in [
                'POLDA', 'POLRES', 'SATPAS', 'METROJAYA', 'METRO JAYA', 'KORLANTAS',
            ]):
                extracted_data['Penerbit'] = row_text
                continue

            # SIM number
            if 'Nomor SIM' not in extracted_data:
                sim_match = re.search(r'(\d{4}-\d{4}-\d{5,6})', row_text)
                if sim_match:
                    extracted_data['Nomor SIM'] = sim_match.group(1)
                else:
                    clean_num = row_text.replace("-", "").replace(" ", "")
                    sim_match2 = re.search(r'(\d{12,16})', clean_num)
                    if sim_match2:
                        extracted_data['Nomor SIM'] = sim_match2.group(1)

            # Section detection
            section_match = re.search(r'\b([1-6])\.', row_text)
            if section_match:
                current_section = int(section_match.group(1))
                clean_val = re.sub(rf'{current_section}\.\s*', '', row_text).strip()
            else:
                clean_val = row_text
                if (current_section == 0 and 'Nomor SIM' in extracted_data
                        and not self.is_garbage(clean_val)
                        and not re.search(r'\d', clean_val)
                        and len(clean_val) > 2):
                    current_section = 1
                if (current_section < 2
                        and re.search(r'\b\d{2}-\d{2}-(19|20)\d{2}\b', clean_val)
                        and clean_val != extracted_data.get('Berlaku Sampai')):
                    current_section = 2
                if current_section < 3 and re.search(
                        r'\b(PRIA|WANITA|LAKI|PEREMPUAN)\b', clean_val.upper()):
                    current_section = 3
                if current_section < 4 and re.search(
                        r'\b(RT|RW|JL|JALAN|GG|GANG|KP|PERUM|GRIYA|KOMP)\b',
                        clean_val.upper()):
                    current_section = 4
                if current_section < 5 and FuzzyMatcher.is_job(clean_val):
                    current_section = 5

            if not clean_val or self.is_garbage(clean_val):
                continue

            if current_section == 1 and len(clean_val) > 2:
                clean_name = re.sub(r'\d+', '', clean_val).strip()
                if clean_name:
                    if 'Nama' not in extracted_data:
                        extracted_data['Nama'] = clean_name
                    else:
                        extracted_data['Nama'] += ' ' + clean_name
            elif current_section == 2:
                if 'Tempat & Tgl. Lahir' not in extracted_data:
                    extracted_data['Tempat & Tgl. Lahir'] = clean_val
                else:
                    extracted_data['Tempat & Tgl. Lahir'] += ' ' + clean_val
            elif current_section == 3:
                match_jk = re.search(
                    r'([ABO]+)\s*[-]*\s*(PRIA|WANITA|LAKI|PEREMPUAN)',
                    clean_val.upper()
                )
                if match_jk:
                    extracted_data['Gol. Darah']   = match_jk.group(1)
                    extracted_data['Jenis Kelamin'] = match_jk.group(2)
                else:
                    extracted_data['Gol. Darah - Kelamin'] = clean_val
            elif current_section == 4:
                if clean_val.replace('.', '').strip() == str(current_section):
                    continue
                address_accumulator.append(clean_val)
            elif current_section == 5:
                if clean_val.replace('.', '').strip() == str(current_section):
                    continue
                if 'Pekerjaan' not in extracted_data:
                    extracted_data['Pekerjaan'] = clean_val
            elif current_section == 6:
                if 'Provinsi' not in extracted_data:
                    extracted_data['Provinsi'] = clean_val

        if address_accumulator:
            extracted_data['raw_address_lines'] = address_accumulator

        return extracted_data


# ---------------------------------------------------------------------------
# Smart SIM strategy (free-form / newer layout)
# ---------------------------------------------------------------------------

class SmartSIMStrategy(BaseSIMStrategy):
    def extract(self, texts: List[str], all_data_with_boxes: List[Dict]) -> Dict[str, Any]:
        extracted_data = {}
        rows      = GeometryUtils.cluster_into_rows(all_data_with_boxes)
        row_texts = [" ".join([x['text'] for x in row]).strip() for row in rows]

        # SIM number
        for t in row_texts:
            clean = t.replace(" ", "").replace("-", "")
            match = re.search(r'(\d{12,16})', clean)
            if match:
                extracted_data['Nomor SIM'] = match.group(1)
                break

        # Expiry
        full_blob    = " ".join(texts)
        dates        = re.findall(r'\b(\d{2})[\s\.-]*(\d{2})[\s\.-]*(20\d{2})\b', full_blob)
        valid_expiry = None
        for d, m, y in dates:
            try:
                if int(y) > 2018:
                    valid_expiry = f"{d}-{m}-{y}"
            except ValueError:
                continue
        if valid_expiry:
            extracted_data['Berlaku Sampai'] = valid_expiry

        # Penerbit
        for t in row_texts:
            if any(p in t.upper() for p in [
                'POLDA', 'POLRES', 'SATPAS', 'METROJAYA', 'METRO JAYA', 'KORLANTAS',
            ]):
                clean_penerbit = re.sub(r'\b\d{2}-\d{2}-20\d{2}\b', '', t).strip()
                if clean_penerbit:
                    extracted_data['Penerbit'] = clean_penerbit
                break

        tagged_rows = [
            {'type': FuzzyMatcher.identify_field(text), 'text': text, 'index': idx}
            for idx, text in enumerate(row_texts)
        ]

        # Nama
        nama_idx = self._find_anchor_index(tagged_rows, 'NAMA')
        if nama_idx is not None:
            val = self._find_value_forward(tagged_rows, nama_idx + 1, 2, ['TTL', 'ALAMAT'])
            if val and not re.search(r'\d', val):
                extracted_data['Nama'] = val
        else:
            if 'Nomor SIM' in extracted_data:
                sim_row_idx = next(
                    (i for i, text in enumerate(row_texts)
                     if extracted_data['Nomor SIM'] in text.replace("-", "").replace(" ", "")),
                    -1
                )
                if sim_row_idx != -1:
                    val = self._find_value_forward(
                        tagged_rows, sim_row_idx + 1, 3, ['TTL', 'ALAMAT']
                    )
                    if val and not re.search(r'\d', val):
                        extracted_data['Nama'] = val

        # TTL
        ttl_idx = self._find_anchor_index(tagged_rows, 'TTL')
        if ttl_idx is not None:
            ttl_raw = self._find_value_forward(
                tagged_rows, ttl_idx + 1, 5, ['GOL_DARAH', 'JK', 'ALAMAT']
            )
            if ttl_raw:
                self._parse_ttl(ttl_raw, extracted_data)
        else:
            for text in row_texts:
                if re.search(r'\b\d{2}-\d{2}-(19|20)\d{2}\b', text):
                    if text != extracted_data.get('Berlaku Sampai'):
                        self._parse_ttl(text, extracted_data)
                        break

        # Jenis Kelamin / Gol. Darah
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
                if 'PRIA' in row.upper() or 'LAKI' in row.upper():
                    extracted_data['Jenis Kelamin'] = 'LAKI-LAKI'
                elif 'WANITA' in row.upper() or 'PEREMPUAN' in row.upper():
                    extracted_data['Jenis Kelamin'] = 'PEREMPUAN'

        # Pekerjaan
        pekerjaan_idx = self._find_anchor_index(tagged_rows, 'PEKERJAAN')
        if pekerjaan_idx is not None:
            val = self._find_value_forward(tagged_rows, pekerjaan_idx + 1, 3, ['PENERBIT'])
            if val and not re.search(r'\b\d{2}-\d{2}-20\d{2}\b', val):
                extracted_data['Pekerjaan'] = val
        else:
            for row in row_texts:
                if FuzzyMatcher.is_job(row) and not self.is_garbage(row):
                    if 'Pekerjaan' not in extracted_data:
                        extracted_data['Pekerjaan'] = row
                        break

        # Alamat
        alamat_idx = self._find_anchor_index(tagged_rows, 'ALAMAT')
        if alamat_idx is not None:
            start    = alamat_idx + 1
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
                if any(p in row.upper() for p in [
                    "SATPAS", "POLRES", "POLDA", "KORLANTAS", "METRO JAYA",
                ]): continue
                if re.search(r'\b\d{2}-\d{2}-20\d{2}\b', row): continue
                if not self.is_garbage(row):
                    addr_lines.append(row)
            extracted_data['raw_address_lines'] = addr_lines

        return extracted_data

    # ------------------------------------------------------------------
    def _find_anchor_index(self, tagged_rows, atype):
        for row in tagged_rows:
            if row['type'] == atype:
                return row['index']
        return None

    def _find_value_forward(self, tagged_rows, start_idx, max_lookahead,
                             stop_types=None):
        limit = min(start_idx + max_lookahead, len(tagged_rows))
        for i in range(start_idx, limit):
            row = tagged_rows[i]
            if stop_types and row['type'] in stop_types: return None
            if self.is_garbage(row['text']): continue
            if len(row['text']) < 3 and not re.search(r'\d', row['text']): continue
            return row['text']
        return None

    def _parse_ttl(self, text: str, data: Dict[str, Any]) -> None:
        """
        Parse a raw Tempat/Tgl Lahir string into 'Tempat Lahir' and
        'Tanggal Lahir'.

        Improvements over the original:
          * Year pattern broadened to ``\\d{2,4}`` so OCR-corrupted years
            like ``4986`` (misread of ``1986``) are passed through
            ``normalize_date_robust`` which attempts single-digit repair.
          * Trailing digit noise is stripped from the extracted place name,
            preventing OCR bleed-through from contaminating ``tempat_lahir``.
        """
        if not text:
            return
        text = text.strip()

        # Broader year pattern — normalize_date_robust handles repair
        date_match = re.search(
            r'(\d{1,2})[\s\-./]+(\d{1,2})[\s\-./]+(\d{2,4})', text
        )
        if date_match:
            d, m, y = date_match.groups()
            raw_date   = f"{d.zfill(2)}-{m.zfill(2)}-{y}"
            normalized = normalize_date_robust(raw_date)

            # Accept only if normalization yielded a proper DD-MM-YYYY string
            if normalized and re.match(r'^\d{2}-\d{2}-\d{4}$', normalized):
                data['Tanggal Lahir'] = normalized

                # Extract place: everything before the date pattern
                place = (
                    text.split(',', 1)[0] if ',' in text
                    else text[:date_match.start()]
                ).strip()

                # Strip any trailing digit noise that bled into the place string
                # e.g. "BOGOR 7" → "BOGOR"
                place = re.sub(r'\s+\d.*$', '', place).strip()
                place = re.sub(r'[,.\s]+$',  '', place).strip()
                if place:
                    data['Tempat Lahir'] = place
                return

        # No parseable date found — try comma split first
        if ',' in text:
            parts = text.split(',', 1)
            data['Tempat Lahir'] = parts[0].strip()
            if len(parts) > 1:
                data['Tanggal Lahir'] = normalize_date_robust(parts[1].strip())
        else:
            # Extract just the alphabetic place name, stripping trailing digits
            place = re.sub(r'\s+\d.*$', '', text).strip()
            data['Tempat Lahir'] = place if len(place) >= 2 else text


# ---------------------------------------------------------------------------
# SIMExtractor
# ---------------------------------------------------------------------------

class SIMExtractor:
    def __init__(self):
        self.legacy_strategy = LegacySIMStrategy()
        self.smart_strategy  = SmartSIMStrategy()

        # Kept for exact/substring fast-path inside _is_region_line()
        self.cities = {
            'JAKARTA', 'BOGOR', 'DEPOK', 'TANGERANG', 'BEKASI', 'BANDUNG',
            'SEMARANG', 'SURABAYA', 'MEDAN', 'MAKASSAR', 'BALIKPAPAN',
            'DENPASAR', 'SLEMAN', 'BANTUL', 'KULON PROGO', 'SERANG',
            'CILEGON', 'CIMAHI', 'SUKABUMI', 'BATAM', 'KUPANG', 'PONOROGO',
            'MALANG', 'SOLO', 'SURAKARTA', 'YOGYAKARTA', 'PALEMBANG',
            'PEKANBARU', 'PADANG', 'LAMPUNG', 'JAMBI', 'BENGKULU', 'ACEH',
            'MATARAM', 'JAYAPURA', 'MANADO', 'AMBON', 'KENDARI', 'PALU',
            'LEBAK', 'PANDEGLANG', 'CIANJUR', 'GARUT', 'TASIKMALAYA', 'CIAMIS',
            'KUNINGAN', 'CIREBON', 'MAJALENGKA', 'SUMEDANG', 'INDRAMAYU',
            'SUBANG', 'PURWAKARTA', 'KARAWANG', 'BANDAR LAMPUNG',
        }

    # ------------------------------------------------------------------
    def detect_version(self, texts: List[str]) -> str:
        full_text = " ".join(texts)
        if re.search(r'\b[1-3]\.\s+(Nama|Tempat|Alamat|Pekerjaan)', full_text, re.IGNORECASE):
            return "LEGACY"
        if re.search(r'\b1\.\s', full_text) and re.search(r'\b2\.\s', full_text):
            return "LEGACY"
        return "SMART"

    # ------------------------------------------------------------------
    def _is_region_line(self, line_u: str) -> bool:
        """
        Decide whether an address line names a city / kabupaten.

        Three-tier check:
          1. Hard structural keywords (KOTA, KAB, KABUPATEN, JAKARTA).
          2. Exact/substring match against the fast ``self.cities`` set.
          3. Fuzzy ``partial_ratio`` ≥ 82 against ``INDONESIAN_REGIONS``.
        """
        if any(kw in line_u for kw in ('KOTA', 'KAB.', 'KAB ', 'KABUPATEN', 'JAKARTA')):
            return True
        if any(c in line_u for c in self.cities):
            return True
        if INDONESIAN_REGIONS:
            _, score = process.extractOne(
                line_u, INDONESIAN_REGIONS, scorer=fuzz.partial_ratio
            )
            return score >= 82
        return False

    # ------------------------------------------------------------------
    def process_sim(self, ocr_result: List[Dict]) -> Optional[Dict[str, Any]]:
        if not ocr_result or not isinstance(ocr_result, list):
            return None
        data = ocr_result[0]
        if not isinstance(data, dict):
            return None

        boxes = data.get('dt_polys',  [])
        texts = data.get('rec_texts', [])
        if not texts or not boxes or len(boxes) != len(texts):
            return None

        all_data = []
        for i, (box, text) in enumerate(zip(boxes, texts)):
            try:
                np_box = np.array(box, dtype=np.int32)
                if np_box.shape != (4, 2):
                    continue
                all_data.append({
                    'id':       i,
                    'box':      np_box,
                    'text':     str(text),
                    'y_center': (np_box[0][1] + np_box[2][1]) / 2,
                })
            except Exception:
                continue

        if not all_data:
            return None

        version  = self.detect_version(texts)
        strategy = self.legacy_strategy if version == "LEGACY" else self.smart_strategy

        try:
            extracted_raw = strategy.extract(texts, all_data)
            final_data    = self.post_process_common(extracted_raw)
            return strategy.cleanup_common(final_data)
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _parse_address_block(self, address_lines: List[str]) -> Dict[str, Optional[str]]:
        addr = {
            "name":      None, "rt_rw":    None, "kel_desa":  None,
            "kecamatan": None, "kabupaten": None, "provinsi":  None,
        }
        if not address_lines:
            return addr

        clean_lines = []
        for line in address_lines:
            line = re.sub(r'^(Alamat|Address)[\s\:\.]*', '', line, flags=re.IGNORECASE).strip()
            line = re.sub(r'^[4]\.\s*', '', line).strip()
            if not line:
                continue
            if FuzzyMatcher.is_job(line):
                continue
            clean_lines.append(line)

        if not clean_lines:
            return addr

        # Identify city / kabupaten line (scan from bottom) using fuzzy region check
        city_index = len(clean_lines)
        for idx in range(len(clean_lines) - 1, -1, -1):
            line_u = clean_lines[idx].upper()
            if self._is_region_line(line_u):
                if not addr['kabupaten']:
                    addr['kabupaten'] = _clean_sim_kabupaten(clean_lines[idx])
                city_index = idx
                break

        street_parts = []
        state        = 0

        rt_pivot_re   = re.compile(r'(?:RT|RW|R\.T|R\.W)[\s\.\:]*(\d{1,4})', re.IGNORECASE)
        rt_sep_re     = re.compile(r'^[\s\/\-\|lI1]+(\d{1,4})', re.IGNORECASE)
        rw_residue_re = re.compile(r'^\s*(?:RW|RW\.|W\.|RW:)[\s\.\:]*(\d{1,4})', re.IGNORECASE)
        street_prefixes = (
            'JL', 'JALAN', 'GG', 'GANG', 'KP', 'KMP', 'KOMP', 'DUSUN',
            'DSN', 'BLK', 'BLOK', 'NO', 'PERUM', 'GRIYA', 'PERUMAHAN',
        )

        for idx, line in enumerate(clean_lines):
            if idx >= city_index:
                break

            line_u = line.upper()

            # Kecamatan
            if 'KEC' in line_u and 'KECIL' not in line_u:
                val = re.sub(r'\b(KEC|KECAMATAN)\b\.?', '', line, flags=re.IGNORECASE).strip()
                addr['kecamatan'] = val
                state = 1
                continue

            # Kel/Desa prefix check
            is_kel_prefix = False
            for p in ['KEL', 'DESA', 'DS']:
                if re.match(rf'^{p}\b', line_u) or re.match(rf'^{p}\.', line_u):
                    is_kel_prefix = True
                    break

            rt_match = rt_pivot_re.search(line)

            if rt_match:
                state     = 1
                start_pos = rt_match.start()
                end_pos   = rt_match.end()
                prefix    = line[:start_pos].strip()
                match_val = rt_match.group(1)
                residue   = line[end_pos:]

                sep_match = rt_sep_re.match(residue)
                rw_val    = None

                if sep_match:
                    rw_val  = sep_match.group(1)
                    residue = residue[sep_match.end():]
                else:
                    rw_match = rw_residue_re.search(residue)
                    if rw_match:
                        rw_val  = rw_match.group(1)
                        residue = residue[rw_match.end():]

                # ── OCR '/' → '1' recovery ────────────────────────────
                # Only triggered when the primary sep-regex found no RW,
                # which prevents false splits on valid RT values like '011'.
                if rw_val is None:
                    recovered = _split_rtrw_ocr(match_val, residue)
                    if recovered:
                        match_val, rw_val, residue = recovered
                # ─────────────────────────────────────────────────────

                if rw_val:
                    addr['rt_rw'] = f"{match_val}/{rw_val}"
                else:
                    addr['rt_rw'] = match_val

                if is_kel_prefix:
                    cleaned_prefix = re.sub(r'\b(KEL|DESA|DS)\b\.?', '', prefix,
                                            flags=re.IGNORECASE).strip()
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
                val = re.sub(r'\b(KEL|DESA|DS)\b\.?', '', line,
                             flags=re.IGNORECASE).strip()
                addr['kel_desa'] = val
                state = 1
                continue

            if state == 0:
                starts_with_street = any(line_u.startswith(p) for p in street_prefixes)
                if ',' in line and not starts_with_street:
                    parts = line.split(',', 1)
                    p1    = parts[0].strip()
                    p2    = parts[1].strip()
                    if not addr['kel_desa']:  addr['kel_desa']  = p1
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

    # ------------------------------------------------------------------
    def post_process_common(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        # Normalise legacy Tempat & Tgl. Lahir field
        if 'Tempat & Tgl. Lahir' in extracted_data:
            val        = extracted_data['Tempat & Tgl. Lahir']
            # Broadened year pattern (same as SmartSIMStrategy._parse_ttl)
            date_match = re.search(
                r'(\d{1,2})[\s\-./]+(\d{1,2})[\s\-./]+(\d{2,4})', val
            )
            if date_match:
                d, m, y  = date_match.groups()
                raw_date = f"{d.zfill(2)}-{m.zfill(2)}-{y}"
                normalized = normalize_date_robust(raw_date)
                if normalized and re.match(r'^\d{2}-\d{2}-\d{4}$', normalized):
                    extracted_data['Tanggal Lahir'] = normalized
                    place = (
                        val.split(',', 1)[0] if ',' in val
                        else val[:date_match.start()]
                    ).strip()
                    place = re.sub(r'\s+\d.*$', '', place).strip()
                    extracted_data['Tempat Lahir'] = place if place else None
                else:
                    if ',' in val:
                        parts = val.split(',', 1)
                        extracted_data['Tempat Lahir'] = parts[0].strip()
                        if len(parts) > 1:
                            extracted_data['Tanggal Lahir'] = normalize_date_robust(
                                parts[1].strip()
                            )
                    else:
                        extracted_data['Tempat Lahir'] = val
            else:
                if ',' in val:
                    parts = val.split(',', 1)
                    extracted_data['Tempat Lahir'] = parts[0].strip()
                    if len(parts) > 1:
                        extracted_data['Tanggal Lahir'] = normalize_date_robust(
                            parts[1].strip()
                        )
                else:
                    extracted_data['Tempat Lahir'] = val
            del extracted_data['Tempat & Tgl. Lahir']

        # Normalise Tanggal Lahir via the authoritative date normalizer
        if extracted_data.get('Tanggal Lahir'):
            extracted_data['Tanggal Lahir'] = normalize_date_robust(
                extracted_data['Tanggal Lahir']
            )

        # ── Clean date residue from Tempat Lahir ─────────────────────────
        # Prevents OCR noise like "BOGOR 7-05 4986" from ending up as the
        # place name when date parsing failed on the first pass.
        if extracted_data.get('Tempat Lahir'):
            tl = extracted_data['Tempat Lahir']
            # Strip "Tgl.Lahir …" label artefacts
            tl = re.sub(r'(?:TGL\.?\s*LAHIR)[.\s:]*\d.*$', '', tl,
                        flags=re.IGNORECASE).strip()
            # Strip trailing digit sequences (date residue)
            tl = re.sub(r'\s+\d.*$', '', tl).strip()
            tl = re.sub(r'[,.\s]+$',  '', tl).strip()
            extracted_data['Tempat Lahir'] = tl if len(tl) >= 2 else None
        # ─────────────────────────────────────────────────────────────────

        # Build address block
        if 'raw_address_lines' in extracted_data:
            parsed = self._parse_address_block(extracted_data['raw_address_lines'])
            extracted_data['alamat'] = parsed
            del extracted_data['raw_address_lines']
        else:
            extracted_data.setdefault('alamat', {
                "name": None, "rt_rw": None, "kel_desa": None,
                "kecamatan": None, "kabupaten": None, "provinsi": None,
            })
            if 'Provinsi' in extracted_data:
                extracted_data['alamat']['provinsi'] = extracted_data['Provinsi']

        return extracted_data

    # ------------------------------------------------------------------
    def calculate_sim_completeness(self, data):
        if not data: return 0.0
        score = 0.0
        if data.get('Nama'):               score += 1.5
        if data.get('Nomor SIM'):          score += 1.0
        if data.get('Tanggal Lahir'):      score += 1.0
        addr = data.get('alamat') or {}
        if addr.get('kabupaten') or addr.get('name'): score += 1.0
        if addr.get('kel_desa'):           score += 0.5
        if data.get('Pekerjaan'):          score += 0.5
        if data.get('Berlaku Sampai'):     score += 0.5
        return score


# ---------------------------------------------------------------------------
# JSON output formatter
# ---------------------------------------------------------------------------

def format_sim_to_json(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not data:
        return {"status": 400, "error": True, "message": "Failed to extract SIM data"}

    addr = data.get('alamat') or {}
    if not isinstance(addr, dict):
        addr = {}

    tgl_raw = data.get("Tanggal Lahir") or None

    return {
        "status":  200,
        "error":   False,
        "message": "SIM OCR Processed Successfully",
        "data": {
            "document_type":     "SIM",
            "nomor":             data.get("Nomor SIM"),
            "nama":              data.get("Nama"),
            "tempat_lahir":      data.get("Tempat Lahir"),
            "tgl_lahir":         normalize_date_robust(tgl_raw) if tgl_raw else None,
            "jenis_kelamin":     data.get("Jenis Kelamin"),
            "agama":             None,
            "status_perkawinan": None,
            "pekerjaan":         data.get("Pekerjaan"),
            "kewarganegaraan":   None,
            "alamat": {
                "name":      addr.get("name"),
                "rt_rw":     addr.get("rt_rw"),
                "kel_desa":  addr.get("kel_desa"),
                "kecamatan": addr.get("kecamatan"),
                "kabupaten": addr.get("kabupaten"),
                "provinsi":  addr.get("provinsi"),
            },
        }
    }