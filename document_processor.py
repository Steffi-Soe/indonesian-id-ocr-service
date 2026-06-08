import cv2
import sys
import os
import traceback
import logging
import numpy as np
from typing import Optional, Dict, Any
from paddleocr import PaddleOCR

from ktp_extractor      import KTPExtractor, format_to_target_json
from sim_extractor       import SIMExtractor, format_sim_to_json
from image_preprocessor import StandardPreprocessor, SmartSIMPreprocessor
from nik_fuzzy           import NIKFuzzyExtractor
from date_normalizer     import DateNormalizer
from confidence_scorer   import KTPConfidenceScorer, print_report
from nik_cross_validator import NIKCrossValidator

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


# ---------------------------------------------------------------------------
# Document-type detection
# ---------------------------------------------------------------------------

import re


def identify_document_type(ocr_texts: list) -> str:
    raw_joined = " ".join(ocr_texts)
    full_text  = raw_joined.upper()
    compact    = re.sub(r'\s+', '', full_text)

    sim_score = ktp_score = 0

    if "SURAT IZIN MENGEMUDI" in full_text: sim_score += 6
    if "DRIVING LICENSE"       in full_text: sim_score += 6
    if "BERLAKU SAMPAI"        in full_text: sim_score += 4
    if "KORLANTAS"             in full_text: sim_score += 3
    if re.search(r'\d{4}[-\s]\d{4}[-\s]\d{5,6}', full_text): sim_score += 4
    for kw in ["SATPAS", "NOMOR SIM", "NO. SIM", "NO SIM"]:
        if kw in full_text: sim_score += 2
    for kw in ["POLDA", "POLRES", "METRO JAYA", "METROJAYA"]:
        if kw in full_text: sim_score += 1
    if re.search(r'\b[1-6]\.\s+[A-Z]', full_text): sim_score += 2

    if "KARTU TANDA PENDUDUK"  in full_text: ktp_score += 6
    if "KEWARGANEGARAAN"       in full_text: ktp_score += 4
    if "STATUS PERKAWINAN"     in full_text: ktp_score += 4
    if "BERLAKU HINGGA"        in full_text: ktp_score += 3
    if re.search(r'\b\d{16}\b', compact):   ktp_score += 5
    for kw in ["PROVINSI", "KABUPATEN", "KECAMATAN"]:
        if kw in full_text: ktp_score += 2
    if re.search(r'\bNIK\b', full_text):    ktp_score += 3
    for kw in ["KEL/DESA", "KEL./DESA", "RT/RW", "GOL. DARAH"]:
        if kw in full_text: ktp_score += 1

    if sim_score > ktp_score and sim_score >= 2: return "SIM"
    if ktp_score >= 2:                           return "KTP"
    if re.search(r'\d{16}', compact):            return "KTP"
    return "UNKNOWN"


def calculate_ocr_confidence(ocr_result: list) -> float:
    if not ocr_result or not ocr_result[0]:
        return 0.0
    scores = ocr_result[0].get("rec_scores", [])
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Post-processor: NIK fuzzy repair + date normalization
# ---------------------------------------------------------------------------

class KTPPostProcessor:
    """
    Applies NIK fuzzy repair and date normalization to an already-extracted
    KTP data dict.  Works on the raw extractor output (before format_to_target_json).

    NIK repair flow (three-tier):
      1. ktp_extractor.cleanup_data() already ran clean_nik_robust(); if that
         yielded a valid 16-digit NIK we skip repair entirely.
      2. If cleanup_data() stored a non-16-digit raw value, best_candidate()
         tries char-substitution + 15→16 padding.
      3. If still unresolved, extract_from_ocr_items() scans every OCR text
         block spatially (anchored to the NIK label y-position).
    """

    def __init__(self):
        self.nik_extractor   = NIKFuzzyExtractor()
        self.date_normalizer = DateNormalizer()

    def repair(
        self, data: Dict[str, Any], ocr_items: list = None
    ) -> Dict[str, Any]:
        if not data:
            return data

        repaired = dict(data)

        # ---- NIK repair ----
        raw_nik = repaired.get("NIK")
        if not raw_nik or not re.match(r'^\d{16}$', str(raw_nik)):
            candidate = None

            if raw_nik:
                candidate = self.nik_extractor.best_candidate(
                    str(raw_nik), min_confidence=0.30
                )

            if candidate is None and ocr_items:
                nik_y = None
                for item in ocr_items:
                    if re.search(r'\bNIK\b', item['text'].upper()):
                        nik_y = (item['box'][0][1] + item['box'][2][1]) / 2
                        break
                candidate = self.nik_extractor.extract_from_ocr_items(
                    ocr_items, nik_y_hint=nik_y
                )

            if candidate:
                logger.info(
                    "NIK repaired: '%s' → '%s' (conf=%.2f, src=%s)",
                    raw_nik, candidate.value, candidate.confidence, candidate.source,
                )
                repaired["NIK"] = candidate.value
                repaired["NIK_confidence"] = candidate.confidence
            else:
                logger.warning("NIK could not be repaired from '%s'", raw_nik)
                repaired["NIK_confidence"] = 0.0
        else:
            repaired["NIK_confidence"] = 1.0

        # ---- Tempat/Tgl Lahir repair ----
        raw_ttl = repaired.get("Tempat/Tgl Lahir", "")
        if raw_ttl:
            place, date_result = self.date_normalizer.normalize_place_date(raw_ttl)
            if date_result.normalized and date_result.confidence > 0.25:
                ttl_new = f"{place},{date_result.normalized}" if place else date_result.normalized
                repaired["Tempat/Tgl Lahir"] = ttl_new
                repaired["TTL_confidence"] = date_result.confidence
            else:
                repaired["TTL_confidence"] = 0.0

        return repaired


# ---------------------------------------------------------------------------
# Main Processor
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """
    KTP / SIM document processor.

    KTP pipeline:
      1. Orientation correction (portrait → landscape via face detection)
      2. Resize to 1000 px wide + white border  (only non-destructive ops)
      3. OCR on the original image
      4. Field extraction (KTPExtractor)
      5. NIK & date repair  (KTPPostProcessor)
      6. Bidirectional NIK ↔ field cross-validation (NIKCrossValidator)
      7. Format to JSON + confidence scoring

    No deskewing, no perspective warping, no adaptive multi-variant OCR,
    no CLAHE/sharpening/denoising.  The original pixel data reaches the
    OCR engine intact.
    """

    def __init__(self, debug: bool = False):
        logger.info("Initialising PaddleOCR engine…")
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='id',
            enable_mkldnn=True,
        )

        self.ktp_extractor = KTPExtractor()
        self.sim_extractor = SIMExtractor()

        self.debug     = debug
        self.debug_dir = "debug_output"
        os.makedirs(self.debug_dir, exist_ok=True)

        # Preprocessors — used only for orientation + resize (KTP)
        # and full preprocessing (SIM, where quality is more variable)
        self.std_preprocessor   = StandardPreprocessor(
            debug=debug, debug_dir=f"{self.debug_dir}/preprocess_std"
        )
        self.smart_preprocessor = SmartSIMPreprocessor(
            debug=debug, debug_dir=f"{self.debug_dir}/preprocess_smart"
        )

        self.ktp_post       = KTPPostProcessor()
        self.cross_validator = NIKCrossValidator()
        self.scorer         = KTPConfidenceScorer()

        logger.info("DocumentProcessor ready.")
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # SIM helpers
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

    def merge_sim_data(self, primary_data, fallback_data):
        if not primary_data: return fallback_data
        if not fallback_data: return primary_data
        merged = primary_data.copy()
        for key in ['Nama', 'Nomor SIM', 'Tempat Lahir', 'Tanggal Lahir',
                    'Jenis Kelamin', 'Pekerjaan', 'Berlaku Sampai']:
            if not merged.get(key) and fallback_data.get(key):
                merged[key] = fallback_data[key]
        addr_prim = merged.get('alamat') or {}
        addr_fall = fallback_data.get('alamat') or {}
        merged_addr = dict(addr_prim)
        for k in ['name', 'rt_rw', 'kel_desa', 'kecamatan', 'kabupaten', 'provinsi']:
            if not merged_addr.get(k) and addr_fall.get(k):
                merged_addr[k] = addr_fall[k]
        merged['alamat'] = merged_addr
        return merged

    # ------------------------------------------------------------------

    def _run_ocr(self, image):
        try:
            result = self.ocr.predict(image)
            conf   = calculate_ocr_confidence(result)
            return result, conf
        except Exception as e:
            logger.warning("OCR failed: %s", e)
            return None, 0.0

    def _get_texts(self, ocr_result):
        if not ocr_result or not ocr_result[0]:
            return []
        return ocr_result[0].get("rec_texts", [])

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_image(self, image_path: str) -> Dict[str, Any]:
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"status": 404, "error": True, "message": "Image not found"}

            # =========================================================
            # PASS 1 — Orientation correction only (portrait → landscape)
            # =========================================================
            oriented = self.std_preprocessor.correct_orientation_semantic(image)

            # =========================================================
            # PASS 2 — Quick OCR for document-type detection
            #          (resize only — no other preprocessing)
            # =========================================================
            quick_img = self.std_preprocessor.add_padding(
                self.std_preprocessor.resize_keep_aspect(oriented, 1000)
            )
            quick_ocr, _ = self._run_ocr(quick_img)
            doc_type = identify_document_type(self._get_texts(quick_ocr))

            if doc_type == "UNKNOWN":
                logger.info("Quick-pass UNKNOWN; retrying on raw image.")
                raw_ocr, _ = self._run_ocr(image)
                raw_type   = identify_document_type(self._get_texts(raw_ocr))
                if raw_type != "UNKNOWN":
                    doc_type  = raw_type
                    quick_ocr = raw_ocr
                    oriented  = image

            sys.stdout.flush()

            if doc_type == "KTP":
                return self._process_ktp(oriented, quick_ocr)

            if doc_type == "SIM":
                return self._process_sim(image, oriented, quick_ocr)

            return {"status": 400, "error": True, "message": "Unknown document type"}

        except Exception as e:
            traceback.print_exc()
            return {"status": 500, "error": True, "message": f"Internal Error: {str(e)}"}

    # ------------------------------------------------------------------
    # KTP processing
    # ------------------------------------------------------------------

    def _process_ktp(
        self,
        oriented_image: np.ndarray,
        initial_ocr: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        KTP pipeline (v3):
          resize + white border  →  OCR on original  →  extract  →  repair
          →  cross-validate NIK ↔ fields  →  format  →  score

        Preprocessing is intentionally minimal:
          * resize_keep_aspect(1000) brings the image to a standard width
          * add_padding(20) adds a white border to prevent OCR edge-clipping
          * No geometric correction, no deskew, no image enhancement
        """
        # ---- Step A: Minimal resize + border (non-destructive) ----
        work_image = self.std_preprocessor.add_padding(
            self.std_preprocessor.resize_keep_aspect(oriented_image, 1000)
        )

        # ---- Step B: OCR on original image ----
        # Re-use the quick-pass result if it came from the same image,
        # otherwise run a fresh prediction on work_image.
        if initial_ocr is not None:
            ocr_result = initial_ocr
            ocr_conf   = calculate_ocr_confidence(ocr_result)
        else:
            ocr_result, ocr_conf = self._run_ocr(work_image)

        if not ocr_result:
            return {"status": 500, "error": True, "message": "OCR produced no result"}

        # ---- Step C: Field extraction ----
        raw_data = self.ktp_extractor.process_ktp(ocr_result, return_trace=False)

        # ---- Step D: NIK fuzzy repair + date normalization ----
        ocr_items     = self._build_ocr_items(ocr_result)
        repaired_data = self.ktp_post.repair(raw_data, ocr_items=ocr_items)

        # ---- Step E: Bidirectional NIK ↔ field cross-validation ----
        repaired_data = self.cross_validator.validate_and_repair(repaired_data)
        cross_val     = repaired_data.pop("_cross_val", None)

        # ---- Step F: Format to JSON ----
        json_output = format_to_target_json(repaired_data)

        # ---- Step G: Confidence scoring ----
        report = self.scorer.score(json_output.get("data", {}))
        if self.debug:
            print_report(report)

        # Apply the cross-validation delta to the composite score
        cv_delta     = getattr(cross_val, "confidence_delta", 0.0) if cross_val else 0.0
        final_score  = max(0.0, min(1.0, report.overall + cv_delta))

        # json_output["confidence"] = {
        #     "overall":          round(final_score, 4),
        #     "grade":            self.scorer._grade(
        #                             final_score,
        #                             report.missing_critical,
        #                         ),
        #     "fields_extracted": report.field_count,
        #     "max_fields":       report.max_field_count,
        #     "cross_check":      report.cross_check_passed,
        #     "missing_critical": report.missing_critical,
        #     "low_confidence":   report.low_confidence,
        #     "ocr_conf":         round(ocr_conf, 4),
        #     "preprocessing":    "minimal",    # documents that no aggressive preprocessing was applied
        #     "ocr_source":       "original",   # OCR ran on the original (resized) image
        #     "nik_structural_bonus": round(report.nik_structural_bonus, 4),
        #     "cross_validation": {
        #         "nik_corrected":    getattr(cross_val, "nik_corrected",    False),
        #         "gender_corrected": getattr(cross_val, "gender_corrected", False),
        #         "date_corrected":   getattr(cross_val, "date_corrected",   False),
        #         "confirmations":    getattr(cross_val, "confirmations",    []),
        #         "conflicts":        getattr(cross_val, "conflicts",        []),
        #         "confidence_delta": round(cv_delta, 4),
        #     },
        # }

        # logger.info(
        #     "KTP processed: grade=%s overall=%.3f fields=%d/%d ocr_conf=%.3f cv_delta=%+.3f",
        #     json_output["confidence"]["grade"],
        #     final_score,
        #     report.field_count,
        #     report.max_field_count,
        #     ocr_conf,
        #     cv_delta,
        # )
        return json_output

    # ------------------------------------------------------------------
    # SIM processing (unchanged)
    # ------------------------------------------------------------------

    def _process_sim(
        self, raw_image, oriented_image, initial_ocr
    ) -> Dict[str, Any]:
        std_image = self.std_preprocessor.add_padding(
            self.std_preprocessor.resize_keep_aspect(oriented_image, 1000)
        )
        ocr_result_std, conf_std = self._run_ocr(std_image)
        if ocr_result_std is None:
            ocr_result_std = initial_ocr
            conf_std       = calculate_ocr_confidence(initial_ocr)

        texts       = self._get_texts(ocr_result_std)
        sim_version = self.sim_extractor.detect_version(texts)
        data_std    = self.sim_extractor.process_sim(ocr_result_std)
        score_std   = self.calculate_sim_completeness(data_std)

        logger.info(
            "SIM: version=%s std_score=%.1f conf=%.2f",
            sim_version, score_std, conf_std,
        )

        if sim_version == "SMART" or score_std < 4.0 or conf_std < 0.70:
            try:
                smart_image = self.smart_preprocessor.preprocess(raw_image)
                ocr_smart, conf_smart = self._run_ocr(smart_image)
                data_smart  = self.sim_extractor.process_sim(ocr_smart)
                score_smart = self.calculate_sim_completeness(data_smart)

                logger.info(
                    "SIM smart path: score=%.1f conf=%.2f", score_smart, conf_smart
                )

                if score_smart >= score_std:
                    final_data = self.merge_sim_data(data_smart, data_std)
                    return format_sim_to_json(final_data)
            except Exception as e:
                logger.error("Smart SIM path failed: %s", e)
                traceback.print_exc()

        return format_sim_to_json(data_std)

    # ------------------------------------------------------------------

    @staticmethod
    def _build_ocr_items(ocr_result) -> list:
        """Convert PaddleOCR output to the list-of-dicts expected by NIKFuzzyExtractor."""
        if not ocr_result or not ocr_result[0]:
            return []
        data   = ocr_result[0]
        boxes  = data.get('dt_polys',   [])
        texts  = data.get('rec_texts',  [])
        scores = data.get('rec_scores', [])
        items  = []
        for i, (box, text) in enumerate(zip(boxes, texts)):
            items.append({
                'id':         i,
                'box':        np.array(box).astype(np.int32),
                'text':       text,
                'confidence': scores[i] if i < len(scores) else 0.0,
            })
        return items