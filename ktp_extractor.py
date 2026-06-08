import re
import numpy as np
from typing import Dict, List, Optional
from thefuzz import process, fuzz

from date_normalizer import normalize_date_robust as _normalize_date  # DD-MM-YYYY, authoritative
from nik_fuzzy import OCR_TO_DIGIT as _NIK_OCR_MAP                   # char-substitution table
from ocr_corrector import OCRTextCorrector as _OCRTextCorrector

# Module-level singleton — used in format_to_target_json for place correction.
_ktp_place_corrector = _OCRTextCorrector()


# ---------------------------------------------------------------------------
# Canonical normalization maps
# ---------------------------------------------------------------------------

# Each key is the canonical (output) form; values are OCR aliases that map to it.
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
                             # common OCR misreads of BURUH
                             "CURLH HARIAN LEPAS", "CURLH HARIAN", "CURUH HARIAN LEPAS",
                             "DURUH HARIAN LEPAS"],
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

KEWARGANEGARAAN_CANONICAL: Dict[str, List[str]] = {
    "WNI": ["WNI", "WN", "WNl", "WN1", "WNI.", "WARGANEGARA INDONESIA", "INDONESIA"],
    "WNA": ["WNA", "WARGANEGARA ASING", "ASING"],
}

STATUS_PERKAWINAN_CANONICAL: Dict[str, List[str]] = {
    "BELUM KAWIN": ["BELUM KAWIN", "BELUM MENIKAH", "SINGLE", "LAJANG",
                    "BLM KAWIN", "BELUMKAWIN"],
    "KAWIN":       ["KAWIN", "MENIKAH", "MARRIED", "SUDAH MENIKAH", "SDH KAWIN"],
    "CERAI HIDUP": ["CERAI HIDUP", "CERAI", "DIVORCED"],
    "CERAI MATI":  ["CERAI MATI", "JANDA", "DUDA"],
}


# ---------------------------------------------------------------------------
# Fuzzy normalization helper
# ---------------------------------------------------------------------------

def _fuzzy_normalize_field(
    value: str,
    canonical_map: Dict[str, List[str]],
    threshold: int = 72,
) -> str:
    """
    Normalize a raw OCR string to its canonical form.

    Algorithm:
      1. Exact (case-insensitive) match against every alias → instant return.
      2. ``fuzz.token_set_ratio`` fuzzy scan across all aliases; return the
         canonical key whose best alias reaches *threshold*.
      3. Return the original value unchanged if nothing qualifies.
    """
    if not value:
        return value

    v_upper = value.upper().strip()

    # Pass 1 – exact alias match
    for canonical, aliases in canonical_map.items():
        if v_upper in [a.upper() for a in aliases]:
            return canonical

    # Pass 2 – fuzzy scan
    best_canonical = value
    best_score     = 0
    for canonical, aliases in canonical_map.items():
        for alias in aliases:
            score = fuzz.token_set_ratio(v_upper, alias.upper())
            if score > best_score:
                best_score     = score
                best_canonical = canonical

    return best_canonical if best_score >= threshold else value


# ---------------------------------------------------------------------------
# Field-level cleaning helpers
# ---------------------------------------------------------------------------

def _clean_nik(raw: str):
    """Extract exactly 16 digits from an OCR'd NIK string.

    Applies the shared OCR character-substitution table (L→1, O→0, etc.)
    before stripping non-digit characters, so a raw read like ``80L112...``
    is treated as ``8011...`` rather than discarded.

    Returns a 16-character digit string or None.
    """
    if not raw:
        return None
    # Apply OCR substitution first so characters like L, O, S are cleaned
    substituted = ''.join(_NIK_OCR_MAP.get(c, c) for c in raw)
    digits = re.sub(r'\D', '', substituted)
    if len(digits) == 16:
        return digits
    if len(digits) > 16:
        m = re.search(r'\d{16}', substituted.replace(' ', ''))
        return m.group(0) if m else None
    return None


def _clean_kabupaten(raw: str) -> str:
    """Remove short leading OCR artefacts before KOTA/KABUPATEN/JAKARTA."""
    if not raw:
        return raw
    cleaned = re.sub(
        r'^[A-Z]{1,4}\s+(?=KOTA\b|KAB\b|KABUPATEN\b|JAKARTA\b)',
        '', raw.strip()
    )
    return cleaned.strip()


def _clean_short_garbage(value: str, min_len: int = 3) -> str:
    """Return empty string if value is suspiciously short."""
    if value and len(value.strip()) < min_len:
        return ''
    return value


# ---------------------------------------------------------------------------
# KTPExtractor
# ---------------------------------------------------------------------------

class KTPExtractor:
    def __init__(self):
        self.canonical_fields = [
            "PROVINSI", "KABUPATEN", "NIK", "Nama", "Tempat/Tgl Lahir",
            "Jenis Kelamin", "Gol. Darah", "Alamat", "RT/RW", "Kel/Desa",
            "Kecamatan", "Agama", "Status Perkawinan", "Pekerjaan",
            "Kewarganegaraan", "Berlaku Hingga"
        ]

        self.truncated_key_map = {
            # RT/RW variants
            "RTIRW":            "RT/RW",
            "RTRW":             "RT/RW",
            "RT.RW":            "RT/RW",
            # Jenis Kelamin variants
            "NIS KELAMIN":      "Jenis Kelamin",
            "ENIS KELAMIN":     "Jenis Kelamin",
            # Tempat/Tgl Lahir variants  ← new entries fix "Tempat/Igliahir" etc.
            "TEMPAT/TGL":       "Tempat/Tgl Lahir",
            "TEMPAT/":          "Tempat/Tgl Lahir",   # catches Tempat/Igliahir
            "EMPAT/TGL":        "Tempat/Tgl Lahir",
            "MPAT/TGL":         "Tempat/Tgl Lahir",
            "TGL LAHIR":        "Tempat/Tgl Lahir",
            "TGL. LAHIR":       "Tempat/Tgl Lahir",
            # Agama variants
            "GAMA":             "Agama",
            # Pekerjaan variants
            "KERJAAN":          "Pekerjaan",
            # Status Perkawinan variants
            "ATUS PERKAWINAN":  "Status Perkawinan",
            # Kel/Desa variants
            "KAL/DESA":         "Kel/Desa",
            "KEL/DESA":         "Kel/Desa",
            # Kecamatan OCR misreads  ← new: catches "Kacamalan", "Kacamatan"
            "KACAMATAN":        "Kecamatan",
            "KACAMALAN":        "Kecamatan",
            "ECAMATAN":         "Kecamatan",
            # NIK alias
            "NO KTP":           "NIK",
            # Nama OCR misread: space inserted into "Nama" → "Na na"
            "NA NA":            "Nama",
        }

        self.known_values = {
            "Agama": [
                "ISLAM", "KRISTEN", "KATOLIK", "HINDU", "BUDDHA", "KONGHUCU",
                "CHRISTIAN", "CATHOLIC",
            ],
            "Jenis Kelamin": [
                "LAKI-LAKI", "PEREMPUAN", "LAKI", "MALE", "FEMALE",
            ],
            "Status Perkawinan": [
                "BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI",
                "MARRIED", "SINGLE", "DIVORCED",
            ],
            # Expanded to include OCR variants so recovery logic can find them
            "Kewarganegaraan": ["WNI", "WNA", "WN", "WARGANEGARA"],
        }

    # ------------------------------------------------------------------
    def _get_y_center(self, item):
        box = item['box']
        return (box[0][1] + box[3][1]) / 2

    # ------------------------------------------------------------------
    def process_ktp(self, ocr_result, return_trace=False):
        if not ocr_result or not ocr_result[0]:
            return (None, None, None) if return_trace else None

        result_dict = ocr_result[0]
        if not result_dict or not isinstance(result_dict, dict):
            return (None, None, None) if return_trace else None

        boxes  = result_dict.get('dt_polys',   [])
        texts  = result_dict.get('rec_texts',  [])
        scores = result_dict.get('rec_scores', [])

        if not texts:
            return (None, None, None) if return_trace else None

        recognized_data = []
        for i, (box, text) in enumerate(zip(boxes, texts)):
            conf = scores[i] if i < len(scores) else 0.0
            recognized_data.append({
                'id':         i,
                'box':        np.array(box).astype(np.int32),
                'text':       text,
                'confidence': conf,
            })

        filtered_data = self.filter_spatial_outliers(recognized_data)
        structured_data, trace_info = self.post_process(filtered_data)
        cleaned_data = self.cleanup_data(structured_data)

        if return_trace:
            return cleaned_data, filtered_data, trace_info
        return cleaned_data

    # ------------------------------------------------------------------
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
        cutoff_y_bottom = max_y + (active_height * 0.45)
        cutoff_y_top    = min_y - (active_height * 0.3)

        return [
            item for item in recognized_data
            if cutoff_y_top <= self._get_y_center(item) <= cutoff_y_bottom
        ]

    # ------------------------------------------------------------------
    def post_process(self, recognized_data):
        potential_keys   = []
        potential_values = []
        trace_info = {}

        for item in recognized_data:
            text_raw   = item['text'].strip()
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
        key_ids = {k['id'] for k in potential_keys}
        key_map = {k['canonical_field']: k for k in potential_keys}

        extracted_data    = {}
        claimed_value_ids = set()

        for key_item in potential_keys:
            key_name = key_item['canonical_field']

            if key_name in extracted_data:
                continue

            if key_name in ["PROVINSI", "KABUPATEN"]:
                raw_text = key_item['text'].strip()
                value = re.sub(
                    re.escape(key_name), '', raw_text, flags=re.IGNORECASE
                ).strip()
                value = re.sub(r'^[:\-\.\s]+', '', value).strip()

                # Fuzzy fallback: exact regex fails for OCR variants like
                # "PRCVINSI" (instead of "PROVINSI").  If the strip produced
                # no change, try removing the first whitespace token when it
                # approximately matches the key name (ratio ≥ 65).
                if not value or value.upper() == raw_text.upper():
                    words = raw_text.split(None, 1)
                    if len(words) == 2 and fuzz.ratio(words[0].upper(), key_name) >= 65:
                        value = re.sub(r'^[:\-\.\s]+', '', words[1]).strip()

                if value:
                    extracted_data[key_name] = value
                    trace_info[key_name] = {
                        "value": value, "source_ids": [key_item['id']],
                        "method": "header_strip"
                    }
                    continue

            key_part_match = process.extractOne(
                key_name, [key_item['text']], scorer=fuzz.partial_ratio
            )
            inline_candidate = ""
            if key_part_match and key_part_match[1] > 70:
                clean_key_text = key_item['text']
                parts = re.split(r'[:]', clean_key_text, maxsplit=1)
                if len(parts) > 1 and parts[1].strip():
                    inline_candidate = parts[1].strip()
                else:
                    if len(clean_key_text) > len(key_name) + 2:
                        potential_inline = clean_key_text[len(key_name):].strip()
                        if re.match(r'^[:\-\.\s]*', potential_inline):
                            inline_candidate = re.sub(r'^[:\-\.\s]*', '', potential_inline)

            if inline_candidate and len(inline_candidate) > 2:
                extracted_data[key_name] = inline_candidate
                trace_info[key_name] = {
                    "value": inline_candidate, "source_ids": [key_item['id']],
                    "method": "inline_extraction"
                }
                continue

            key_y_center = self._get_y_center(key_item)
            key_x_end    = key_item['box'][1][0]
            same_line_candidates = []
            vertical_threshold = 25

            for val_item in potential_values:
                if val_item['id'] in claimed_value_ids:
                    continue
                val_y_center = self._get_y_center(val_item)
                val_x_start  = val_item['box'][0][0]

                if (abs(val_y_center - key_y_center) < vertical_threshold
                        and val_x_start > (key_x_end - 20)):
                    x_dist = val_x_start - key_x_end
                    y_diff = abs(val_y_center - key_y_center)
                    score  = x_dist + (y_diff * 15)
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
                    used_ids   = [best_candidate['id']]
                    method     = "geometric_match"

                    if key_name == 'Alamat':
                        rt_rw_key  = key_map.get('RT/RW')
                        rt_rw_y    = (self._get_y_center(rt_rw_key)
                                      if rt_rw_key else float('inf'))
                        addr_line1_y = self._get_y_center(best_candidate)

                        second_line_cands = []
                        for val_item in recognized_data:
                            if val_item['id'] in claimed_value_ids:            continue
                            if val_item['id'] == best_candidate['id']:         continue
                            if val_item['id'] == key_item['id']:               continue

                            val_y    = self._get_y_center(val_item)
                            txt_up   = val_item['text'].upper()

                            is_below_addr = val_y > (addr_line1_y + 10)
                            is_above_rtrw = val_y < (rt_rw_y - 10)
                            is_close      = (val_y - addr_line1_y) < 45

                            if is_below_addr and is_above_rtrw and is_close:
                                if val_item['id'] in key_ids:           continue
                                if re.search(r'\d{3}[/\s-]+\d{3}',
                                             val_item['text']):         continue
                                if "RT" in txt_up and "RW" in txt_up:  continue
                                if "KEL/DESA" in txt_up:               continue
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
                        "value": value_text, "source_ids": used_ids,
                        "key_id_used": key_item['id'], "method": method
                    }

            if key_name == "NIK" and key_name not in extracted_data:
                below_candidates = []
                for val_item in potential_values:
                    if val_item['id'] in claimed_value_ids: continue
                    val_y_center = self._get_y_center(val_item)
                    y_diff       = val_y_center - key_y_center
                    if 0 < y_diff < 50:
                        clean_val = val_item['text'].replace(" ", "").replace(":", "")
                        if re.match(r'\d+', clean_val):
                            below_candidates.append(val_item)

                if below_candidates:
                    below_candidates.sort(key=lambda x: x['box'][0][1])
                    best_nik = below_candidates[0]
                    extracted_data["NIK"] = best_nik['text']
                    claimed_value_ids.add(best_nik['id'])
                    trace_info["NIK"] = {
                        "value": best_nik['text'], "source_ids": [best_nik['id']],
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

    # ------------------------------------------------------------------
    def recover_missing_fields(self, extracted, values, claimed_ids,
                               key_map, trace_info):
        for field, keywords in self.known_values.items():
            if field in extracted:
                continue

            for val_item in values:
                if val_item['id'] in claimed_ids:
                    continue

                text_upper = val_item['text'].upper()

                if field == "Jenis Kelamin":
                    if "LAKILAKI" in text_upper:
                        extracted[field] = "LAKI-LAKI"
                        claimed_ids.add(val_item['id'])
                        trace_info[field] = {
                            "value": "LAKI-LAKI", "source_ids": [val_item['id']],
                            "method": "typo_recovery"
                        }
                        break

                if field == "Status Perkawinan":
                    if re.search(r'\bKAWIN\b', text_upper):
                        extracted[field] = val_item['text'].upper().strip()
                        claimed_ids.add(val_item['id'])
                        trace_info[field] = {
                            "value": extracted[field], "source_ids": [val_item['id']],
                            "method": "regex_kawin_recovery"
                        }
                        break

                match = process.extractOne(
                    text_upper, keywords, scorer=fuzz.token_set_ratio
                )
                if match and match[1] > 85:
                    extracted[field] = val_item['text']
                    claimed_ids.add(val_item['id'])
                    trace_info[field] = {
                        "value": val_item['text'], "source_ids": [val_item['id']],
                        "method": "value_keyword_recovery"
                    }
                    break

        # Tempat/Tgl Lahir recovery via regex
        if "Tempat/Tgl Lahir" not in extracted:
            for val_item in values:
                if val_item['id'] in claimed_ids: continue
                txt = val_item['text']
                if re.search(r'\d{2}[-\s/]\d{2}[-\s/]\d{4}', txt):
                    if re.search(r'[A-Za-z]{3,}', txt):
                        extracted["Tempat/Tgl Lahir"] = txt
                        claimed_ids.add(val_item['id'])
                        trace_info["Tempat/Tgl Lahir"] = {
                            "value": txt, "source_ids": [val_item['id']],
                            "method": "regex_date_place_recovery"
                        }
                        break

        # Nama recovery via positional inference
        if "Nama" not in extracted:
            nik_key = key_map.get("NIK")
            ttl_key = key_map.get("Tempat/Tgl Lahir")

            y_min = -1
            y_max = float('inf')
            if nik_key:
                y_min = nik_key['box'][3][1]
            if ttl_key:
                y_max = ttl_key['box'][0][1]

            candidates = []
            for val_item in values:
                if val_item['id'] in claimed_ids: continue
                y_center = self._get_y_center(val_item)
                valid = False
                if y_min != -1 and y_max != float('inf'):
                    if y_min < y_center < y_max: valid = True
                elif y_min != -1:
                    if y_min < y_center < y_min + 70: valid = True
                elif y_max != float('inf'):
                    if y_max - 70 < y_center < y_max: valid = True

                if valid:
                    candidates.append(val_item)

            if candidates:
                candidates.sort(key=lambda c: c['box'][0][0])
                chosen = candidates[0]
                extracted["Nama"] = chosen['text']
                claimed_ids.add(chosen['id'])
                trace_info["Nama"] = {
                    "value": chosen['text'], "source_ids": [chosen['id']],
                    "method": "positional_inference_name"
                }

        # NIK recovery via 16-digit regex
        if "NIK" not in extracted:
            for val_item in values:
                if val_item['id'] in claimed_ids: continue
                clean_text = val_item['text'].replace(" ", "").strip()
                if re.match(r'^\d{16}$', clean_text):
                    extracted["NIK"] = clean_text
                    claimed_ids.add(val_item['id'])
                    trace_info["NIK"] = {
                        "value": clean_text, "source_ids": [val_item['id']],
                        "method": "regex_recovery_16_digits"
                    }
                    break

    # ------------------------------------------------------------------
    def cleanup_data(self, data):
        """Normalise extracted field values and fix known OCR error patterns."""
        if not data:
            return data

        cleaned_data = {}

        for key, value in data.items():
            if value is None:
                continue

            clean_value = str(value).strip()
            if clean_value.startswith(':'):
                clean_value = clean_value[1:].strip()

            # ---- NIK ---------------------------------------------------
            if key == "NIK":
                validated = _clean_nik(clean_value)
                if validated:
                    clean_value = validated
                else:
                    # _clean_nik failed: not 16 clean digits even after OCR
                    # substitution.  Rather than dropping the field entirely,
                    # keep the substituted digit-only string so that
                    # KTPPostProcessor.repair() can attempt 15→16 padding.
                    # Discard only when there is clearly no digit content.
                    subst_nik  = ''.join(_NIK_OCR_MAP.get(c, c) for c in clean_value)
                    digits_nik = re.sub(r'\D', '', subst_nik)
                    if len(digits_nik) < 12:
                        continue
                    clean_value = digits_nik   # e.g. "801112303920003" (15 digits)

            # ---- Agama -------------------------------------------------
            elif key == "Agama":
                match, score = process.extractOne(
                    clean_value.upper(), self.known_values['Agama']
                )
                if score > 70:
                    clean_value = match

            # ---- RT/RW -------------------------------------------------
            elif key == "RT/RW":
                if not re.search(r'\d', clean_value):
                    continue
                nums = re.findall(r'\d+', clean_value)
                if len(nums) >= 2:
                    clean_value = f"{nums[0].zfill(3)}/{nums[1].zfill(3)}"

            # ---- Kel/Desa: detect RT/RW bleed-through ------------------
            elif key == "Kel/Desa":
                if re.match(r'^\d{2,3}/\d{2,3}$', clean_value.strip()):
                    if "RT/RW" not in cleaned_data:
                        nums = re.findall(r'\d+', clean_value)
                        if len(nums) >= 2:
                            cleaned_data["RT/RW"] = f"{nums[0].zfill(3)}/{nums[1].zfill(3)}"
                    continue

            # ---- Jenis Kelamin -----------------------------------------
            elif key == "Jenis Kelamin":
                val_upper = clean_value.upper()
                if "LAKI" in val_upper or "MALE" in val_upper or "LK" in val_upper:
                    clean_value = "LAKI-LAKI"
                elif "PEREMPUAN" in val_upper or "FEMALE" in val_upper or "PR" in val_upper:
                    clean_value = "PEREMPUAN"

            # ---- Status Perkawinan ------------------------------------
            elif key == "Status Perkawinan":
                val_upper = clean_value.upper()

                # Pre-normalize common OCR misreads of "BELUM":
                #   "CEL UM" / "CELUM"  — B misread as C
                #   "SEL UM" / "SELUM"  — B misread as S
                val_upper = re.sub(r'\bCEL\s*UM\b', 'BELUM', val_upper)
                val_upper = re.sub(r'\bSEL\s*UM\b', 'BELUM', val_upper)

                # Layer 1: reliable keyword substrings
                if "BELUM" in val_upper or "SINGLE" in val_upper or "LAJANG" in val_upper:
                    clean_value = "BELUM KAWIN"
                elif re.search(r'KAWIN|MARRIED', val_upper) and "BELUM" not in val_upper:
                    clean_value = "KAWIN"
                elif "CERAI" in val_upper or "DIVORCED" in val_upper:
                    if "HIDUP" in val_upper:
                        clean_value = "CERAI HIDUP"
                    elif "MATI" in val_upper:
                        clean_value = "CERAI MATI"
                    else:
                        clean_value = "CERAI"
                else:
                    # Layer 2: OCR-robust heuristic for "BELUM KAWIN"
                    # e.g. "BELUIERAWIN" starts with BELU and ends with AWIN
                    if val_upper.startswith("BELU") and val_upper.endswith("AWIN"):
                        clean_value = "BELUM KAWIN"
                    else:
                        # Layer 3: fuzzy fallback against canonical map
                        normalized = _fuzzy_normalize_field(
                            clean_value,
                            STATUS_PERKAWINAN_CANONICAL,
                            threshold=65,
                        )
                        if normalized in STATUS_PERKAWINAN_CANONICAL:
                            clean_value = normalized

            # ---- Alamat -----------------------------------------------
            elif key == "Alamat":
                clean_value = re.sub(r'\s+RT.*', '', clean_value, flags=re.IGNORECASE).strip()
                clean_value = re.sub(r'\s+RW.*', '', clean_value, flags=re.IGNORECASE).strip()

            # ---- Pekerjaan --------------------------------------------
            elif key == "Pekerjaan":
                # Step 1: regex-based hardcoded fixes (fast, precise)
                clean_value = clean_value.replace("BURUHHARIAN", "BURUH HARIAN")
                clean_value = re.sub(r'\bDURUH\b', 'BURUH', clean_value, flags=re.IGNORECASE)
                clean_value = re.sub(r'\bCURLH\b', 'BURUH', clean_value, flags=re.IGNORECASE)
                clean_value = re.sub(r'\bCURUH\b', 'BURUH', clean_value, flags=re.IGNORECASE)
                clean_value = re.sub(r'HARIANEEPAS', 'HARIAN LEPAS', clean_value)
                # HARIANLEPAS (no space) and HARIANCEPAS (C misread as L)
                clean_value = re.sub(r'HARIAN\s*[CL]EPAS', 'HARIAN LEPAS', clean_value)
                # Step 2: canonical fuzzy normalization
                normalized = _fuzzy_normalize_field(
                    clean_value, PEKERJAAN_CANONICAL, threshold=72
                )
                if normalized in PEKERJAAN_CANONICAL:
                    clean_value = normalized

            # ---- Kewarganegaraan  (WN → WNI, etc.) -------------------
            elif key == "Kewarganegaraan":
                normalized = _fuzzy_normalize_field(
                    clean_value, KEWARGANEGARAAN_CANONICAL, threshold=80
                )
                if normalized in KEWARGANEGARAAN_CANONICAL:
                    clean_value = normalized

            # ---- KABUPATEN / PROVINSI ---------------------------------
            elif key in ("KABUPATEN", "PROVINSI"):
                clean_value = _clean_kabupaten(clean_value)
                # Restore missing space in "DKIJAKARTA" → "DKI JAKARTA"
                clean_value = re.sub(
                    r'\bDKI\s*JAKARTA\b', 'DKI JAKARTA',
                    clean_value, flags=re.IGNORECASE
                )
                clean_value = _clean_short_garbage(clean_value, min_len=3)
                if not clean_value:
                    continue

            # ---- Kecamatan: collapse spaces, strip trailing punctuation
            elif key == "Kecamatan":
                clean_value = re.sub(r'\s{2,}', ' ', clean_value).strip()
                clean_value = re.sub(r'[,./\s]+$', '', clean_value).strip()

            if not clean_value:
                continue

            cleaned_data[key] = clean_value

        return cleaned_data


# ---------------------------------------------------------------------------
# JSON output formatter
# ---------------------------------------------------------------------------

def format_to_target_json(data):
    tempat_lahir = None
    tgl_lahir    = None

    raw_ttl = data.get("Tempat/Tgl Lahir", "") if data else ""

    if raw_ttl:
        if ',' in raw_ttl:
            parts = raw_ttl.split(',', 1)
            tempat_lahir = parts[0].strip().strip(":.,")
            tgl_lahir    = _normalize_date(parts[1].strip())
        else:
            date_match = re.search(
                r'(?P<date>\d{1,2}[-./\s]+\d{1,2}[-./\s]+\d{2,4})\s*$',
                raw_ttl
            )
            if date_match:
                tempat_lahir = raw_ttl[:date_match.start()].strip().strip(":.,")
                tgl_lahir    = _normalize_date(date_match.group('date').strip())
            else:
                tempat_lahir = raw_ttl.strip().strip(":.,")

    # Fuzzy place-name correction: catches OCR truncations like "EBAK" → "LEBAK"
    # where a leading letter was read as ':' and later stripped.
    # Threshold 0.88 is deliberately conservative (false-positive rate ~0)
    # while still catching clear 1-char prefix drops scored at 0.89+.
    if tempat_lahir:
        _corr, _conf = _ktp_place_corrector.correct_field("tempat_lahir", tempat_lahir)
        if _conf >= 0.88 and _corr != tempat_lahir:
            tempat_lahir = _corr

    return {
        "status":  200,
        "error":   False,
        "message": "KTP OCR Processed Successfully",
        "data": {
            "document_type":    "KTP",
            "nomor":            data.get("NIK")                if data else None,
            "nama":             data.get("Nama")               if data else None,
            "tempat_lahir":     tempat_lahir,
            "tgl_lahir":        tgl_lahir,
            "jenis_kelamin":    data.get("Jenis Kelamin")      if data else None,
            "agama":            data.get("Agama")              if data else None,
            "status_perkawinan": data.get("Status Perkawinan") if data else None,
            "pekerjaan":        data.get("Pekerjaan")          if data else None,
            "kewarganegaraan":  data.get("Kewarganegaraan")    if data else None,
            "alamat": {
                "name":      data.get("Alamat")     if data else None,
                "rt_rw":     data.get("RT/RW")      if data else None,
                "kel_desa":  data.get("Kel/Desa")   if data else None,
                "kecamatan": data.get("Kecamatan")  if data else None,
                "kabupaten": data.get("KABUPATEN")  if data else None,
                "provinsi":  data.get("PROVINSI")   if data else None,
            },
        }
    }