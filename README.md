# indonesian-id-ocr-service

# 🪪 Indonesian Document OCR API (KTP & SIM)

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-API-green)
![OCR](https://img.shields.io/badge/PaddleOCR-Enabled-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

This project provides a **Flask-based REST API** for extracting structured data from Indonesian identity documents, including **ID Cards (KTP)** and  **Driving Licenses (SIM)** , using  **PaddleOCR** .

It automatically identifies the document type, performs Optical Character Recognition (OCR), intelligently processes the text, and formats the results into a clean, standardized JSON structure. The pipeline is designed to be  **robust against real-world OCR noise** , including character substitutions, truncated fields, missing separators, and mobile-capture imperfections.

---

## 🚀 Features

* 🔤 **OCR powered by PaddleOCR (Bahasa Indonesia model)**
* ✅ **Multi-Document Support:** Accurately processes both KTP and SIM cards
* 🔍 **Automatic Document Identification** via keyword scoring
* 🧭 **Orientation correction** using face detection (portrait → landscape)
* 🧠 **Multi-stage post-processing pipeline:**
  * Fuzzy NIK extraction with OCR character substitution (`L→1`, `O→0`, etc.)
  * 15→16 digit NIK reconstruction from partial reads
  * Bidirectional NIK ↔ field cross-validation (date, gender)
  * Robust date normalization with year repair and multi-strategy fallback
  * Place-name fuzzy correction against an Indonesian administrative-area database
* 🧹 **Field normalization** for Pekerjaan, Status Perkawinan, Kewarganegaraan, and more
* 📊 **Per-field confidence scoring** with A–F document grading
* 🐛 **10-stage field-level debugger** with annotated image output
* 📦 **Standardized JSON output** across document types

---

## 🧰 Requirements

Ensure you have Python **3.8+** installed.

### Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ `paddlepaddle` (the backend for PaddleOCR) might need to be installed manually depending on your system:
>
> ```bash
> # For CPU
> pip install paddlepaddle==3.2.0
>
> # For GPU (ensure you have a compatible CUDA version)
> pip install paddlepaddle-gpu==3.2.0
> ```

---

## 📁 Project Structure

```
indonesian-id-ocr-service/
│
├── app.py                    # Flask API entry point — request handling & logging
├── document_processor.py     # Pipeline controller: preprocessing → OCR → extraction → scoring
│
├── ktp_extractor.py          # KTP field extraction, normalization, and JSON formatting
├── sim_extractor.py          # SIM field extraction (legacy & smart layout strategies)
│
├── image_preprocessor.py     # StandardPreprocessor (KTP) + SmartSIMPreprocessor
├── nik_fuzzy.py              # Fuzzy NIK extraction: char substitution + 15→16 reconstruction
├── nik_cross_validator.py    # Bidirectional NIK ↔ demographic field repair
├── date_normalizer.py        # Robust DD-MM-YYYY normalization with year repair
├── confidence_scorer.py      # Per-field scoring, cross-check validation, A–F grading
├── ocr_corrector.py          # Char substitution + fuzzy place-name correction
│
├── debug_extraction.py       # 10-stage field-level KTP extraction debugger
│
├── uploads/                  # Temporary storage for uploaded images
├── ocr_logs/                 # Monthly OCR prediction logs (image + JSON)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore
```

---

## 🧠 How It Works

1. A client sends a `POST /ocr/document` request with an image file as `multipart/form-data`.
2. The Flask server validates and temporarily stores the image.
3. **Orientation correction** — face detection rotates portrait images to landscape.
4. **Minimal preprocessing** — resize to 1000 px wide + white border padding. No sharpening, CLAHE, or deskew; the original pixel data reaches the OCR engine intact.
5. **Document type detection** — keyword scoring distinguishes KTP from SIM.
6. **OCR** via PaddleOCR (Bahasa Indonesia, `use_textline_orientation=True`).
7. **Field extraction** (`KTPExtractor`) — spatial bounding-box alignment, fuzzy key matching, inline and geometric value recovery.
8. **NIK fuzzy repair** (`NIKFuzzyExtractor`) — OCR char substitution, 15→16 digit reconstruction, structural scoring.
9. **Date normalization** (`DateNormalizer`) — multi-strategy parsing, year repair for corrupted 4-digit years (e.g. `1392 → 1992`).
10. **Cross-validation** (`NIKCrossValidator`) — NIK encodes birth date and gender; mismatches are auto-corrected with NIK as ground truth.
11. **Confidence scoring** (`KTPConfidenceScorer`) — per-field scores, NIK structural bonus, composite A–F grade.
12. The API returns a standardized JSON response.

### SIM Pipeline

Follows the same orientation and OCR steps, then routes to either a **Legacy** (numbered-section) or **Smart** (free-form) extraction strategy based on layout detection. A higher-resolution preprocessing path (`SmartSIMPreprocessor`) is used as a fallback for lower-quality captures.

---

## ▶️ Running the Server

```bash
python app.py
```

The API will be available at:

```
http://0.0.0.0:5000
```

The server uses **Waitress** (4 threads, 600 s timeout) in production mode.

---

## 📤 Example API Request

### Endpoint

```
POST /ocr/document
```

### Request Body

* **Type:** `form-data`
* **Key:** `image`
* **Value:** Image file (`jpg`, `jpeg`, or `png`)

### Example cURL

```bash
curl -X POST http://localhost:5000/ocr/document \
     -F "image=@/path/to/your_image.jpg"
```

---

## ✅ Example Responses

### KTP Response

```json
{
    "status": 200,
    "error": false,
    "message": "KTP OCR Processed Successfully",
    "data": {
        "document_type": "KTP",
        "nomor": "3201123456789001",
        "nama": "BUDI SANTOSO",
        "tempat_lahir": "BANDUNG",
        "tgl_lahir": "01-01-1990",
        "jenis_kelamin": "LAKI-LAKI",
        "agama": "ISLAM",
        "status_perkawinan": "KAWIN",
        "pekerjaan": "KARYAWAN SWASTA",
        "kewarganegaraan": "WNI",
        "alamat": {
            "name": "JL. MERDEKA NO. 10",
            "rt_rw": "001/002",
            "kel_desa": "CIHAMPELAS",
            "kecamatan": "CIMAHI",
            "kabupaten": "KABUPATEN BANDUNG",
            "provinsi": "JAWA BARAT"
        }
    }
}
```

### SIM Response

```json
{
    "status": 200,
    "error": false,
    "message": "SIM OCR Processed Successfully",
    "data": {
        "document_type": "SIM",
        "nomor": "1198-8017-000562",
        "nama": "MUHAMMAD YUNUS",
        "tempat_lahir": "JAKARTA",
        "tgl_lahir": "08-10-1998",
        "jenis_kelamin": "LAKI-LAKI",
        "agama": null,
        "status_perkawinan": null,
        "pekerjaan": "PELAJAR/MAHASISWA",
        "kewarganegaraan": null,
        "alamat": {
            "name": "JL. H. OYAR NO. 24 PEGANGSAAN DUA",
            "rt_rw": "002/002",
            "kel_desa": null,
            "kecamatan": "KELAPA GADING",
            "kabupaten": "JAKARTA TIMUR",
            "provinsi": "METRO JAYA"
        }
    }
}
```

---

## 🧩 Key Components

| File                        | Description                                                     |
| --------------------------- | --------------------------------------------------------------- |
| **`app.py`**                | Flask REST API server for handling OCR requests                 |
| **`document_processor.py`** | Core pipeline controller: preprocessing → OCR → extraction      |
| **`image_preprocessor.py`** | Image normalization and correction for skew, rotation, and tilt |
| **`ktp_extractor.py`**      | KTP-specific OCR parsing and field mapping                      |
| **`sim_extractor.py`**      | SIM-specific OCR parsing and field mapping                      |
| **`debug_visualizer.py`**   | Generates OCR bounding box visuals, legends, and trace logs     |
| **`uploads/`**              | Temporary image storage during processing                       |
---

## ⚙️ Customization

| What to change                   | Where                                                         |
| -------------------------------- | ------------------------------------------------------------- |
| OCR char substitution table      | `nik_fuzzy.py`→`OCR_TO_DIGIT`                            |
| Occupation canonical map         | `ktp_extractor.py`→`PEKERJAAN_CANONICAL`                 |
| Marital status / religion values | `ktp_extractor.py`→`STATUS_PERKAWINAN_CANONICAL`         |
| Indonesian place database        | `ocr_corrector.py`→`_PROVINCES`,`_KOTA`,`_KABUPATEN` |
| Field confidence weights         | `confidence_scorer.py`→`FIELD_WEIGHTS`                   |
| NIK validity rules               | `nik_fuzzy.py`→`_validate_structure()`                   |
| SIM layout keywords              | `sim_extractor.py`→`FuzzyMatcher.ANCHORS`                |

---

## 🧪 Troubleshooting

| Issue                             | Possible Cause                             | Recommended Fix                                                        |
| --------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------- |
| NIK shows wrong value             | OCR misread leading digit                  | Check Stage 6 of debugger; verify original image quality               |
| `tempat_lahir`empty             | Date separator swallowed by OCR            | NIK cross-validator auto-injects date; place extracted from prefix     |
| Wrong `status_perkawinan`       | B→C OCR confusion in "BELUM"              | Pre-normalization handles `CEL UM`→`BELUM KAWIN`                  |
| Wrong `pekerjaan`               | `HARIANCEPAS`instead of `HARIAN LEPAS` | Regex covers C→L confusion; fuzzy fallback catches remaining variants |
| Low overall grade                 | Several fields missing or low-confidence   | Run debugger to identify earliest failing stage                        |
| OCR returned no results           | Poor image quality                         | Use a clearer, higher-resolution image with even lighting              |
| Could not determine document type | Non-KTP/SIM image                          | Upload a valid Indonesian identity document                            |
| `paddlepaddle`not found         | Dependency missing                         | Install `paddlepaddle`manually (see Requirements)                    |

---

## 🧑‍💻 Author

**Developed by:** *Steffi Soeroredjo*
📧 Email: [steffisoeroredjo5@gmail.com](mailto:steffisoeroredjo5@gmail.com)
🌐 GitHub: [https://github.com/Steffi-Soe](https://github.com/Steffi-Soe)
