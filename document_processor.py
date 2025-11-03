import cv2
from paddleocr import PaddleOCR
from ktp_extractor import KTPExtractor, format_to_target_json
from sim_extractor import SIMExtractor, format_sim_to_json

def identify_document_type(ocr_texts):
    full_text = " ".join(ocr_texts).upper()

    if "SURAT IZIN MENGEMUDI" in full_text or "DRIVING LICENSE" in full_text:
        return "SIM"
    
    if "PROVINSI" in full_text and "NIK" in full_text:
        return "KTP"

    return "UNKNOWN"


class DocumentProcessor:
    def __init__(self):
        print("Initializing PaddleOCR engine...")
        self.ocr = PaddleOCR(use_textline_orientation=True, lang='id')
        self.ktp_extractor = KTPExtractor()
        self.sim_extractor = SIMExtractor()
        print("Engine ready for processing.")

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return {"status": 404, "error": True, "message": f"Could not read image at {image_path}"}

        ocr_result = self.ocr.predict(image)

        if not ocr_result or not ocr_result[0] or not ocr_result[0]['rec_texts']:
            return {"status": 500, "error": True, "message": "OCR failed to detect any text."}
        
        ocr_texts = ocr_result[0]['rec_texts']
        doc_type = identify_document_type(ocr_texts)

        if doc_type == "KTP":
            ktp_data = self.ktp_extractor.process_ktp(ocr_result)
            if not ktp_data:
                return {"status": 500, "error": True, "message": "Failed to extract KTP data."}
            return format_to_target_json(ktp_data)

        elif doc_type == "SIM":
            sim_data = self.sim_extractor.process_sim(ocr_result)
            if not sim_data:
                return {"status": 500, "error": True, "message": "Failed to extract SIM data."}
            return format_sim_to_json(sim_data)
        
        else:
            return {"status": 400, "error": True, "message": "Could not determine document type (not a KTP or SIM)."}