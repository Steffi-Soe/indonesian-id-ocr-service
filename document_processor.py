import cv2
import numpy as np
import time
from paddleocr import PaddleOCR
from ktp_extractor import KTPExtractor, format_to_target_json
from sim_extractor import SIMExtractor, format_sim_to_json
from image_preprocessor import ImagePreprocessor
from debug_visualizer import DebugVisualizer

def identify_document_type(ocr_texts):
    full_text = " ".join(ocr_texts).upper()

    if "SURAT IZIN MENGEMUDI" in full_text:
        return "SIM"
    if "PROVINSI" in full_text and ("NIK" in full_text or "KARTU TANDA" in full_text):
        return "KTP"
    return "UNKNOWN"

class DocumentProcessor:
    def __init__(self, debug=False):
        print("Initializing PaddleOCR engine...")
        self.ocr = PaddleOCR(use_textline_orientation=True, lang='id', enable_mkldnn=True)

        self.ktp_extractor = KTPExtractor()
        self.sim_extractor = SIMExtractor()
        
        self.debug = debug
        self.debug_dir = "debug_output"
        self.preprocessor = ImagePreprocessor(
            debug=self.debug,
            debug_dir=f"{self.debug_dir}/preprocess"
        )
        self.visualizer = DebugVisualizer(output_dir=f"{self.debug_dir}/analysis")

        print("Engine ready.")

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return {"status": 404, "error": True, "message": "Image not found"}

        ts = int(time.time())
        corrected_image = self.preprocessor.preprocess(image)
        
        ocr_result = self.ocr.predict(corrected_image)
        
        doc_type = "UNKNOWN"
        if ocr_result and ocr_result[0] and ocr_result[0].get("rec_texts"):
            doc_type = identify_document_type(ocr_result[0]["rec_texts"])
            final_image_to_process = corrected_image

        if doc_type == "UNKNOWN":
            print("Preprocessing yielded UNKNOWN type. Attempting fallback to raw image.")
            h, w = image.shape[:2]
            if w > 1500:
                scale = 1500 / w
                image = cv2.resize(image, None, fx=scale, fy=scale)
                
            ocr_result_raw = self.ocr.predict(image)
            if ocr_result_raw and ocr_result_raw[0] and ocr_result_raw[0].get("rec_texts"):
                fallback_type = identify_document_type(ocr_result_raw[0]["rec_texts"])
                if fallback_type != "UNKNOWN":
                    doc_type = fallback_type
                    ocr_result = ocr_result_raw
                    final_image_to_process = image

        if doc_type == "KTP":
            if self.debug:
                data, ocr_data_with_ids, trace = self.ktp_extractor.process_ktp(
                    ocr_result, return_trace=True
                )
                
                if ocr_data_with_ids:
                    self.visualizer.visualize_ocr(
                        final_image_to_process, 
                        ocr_data_with_ids, 
                        prefix=f"{ts}_ktp_ocr"
                    )
                    self.visualizer.save_extraction_trace(trace, prefix=f"{ts}_ktp")
            else:
                data = self.ktp_extractor.process_ktp(
                    ocr_result, return_trace=False
                )

            return format_to_target_json(data)

        if doc_type == "SIM":
            data = self.sim_extractor.process_sim(ocr_result)
            return format_sim_to_json(data)

        return {"status": 400, "error": True, "message": "Unknown document type"}