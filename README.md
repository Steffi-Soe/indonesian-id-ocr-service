# indonesian-id-ocr-service

# 🪪 Indonesian Document OCR API (KTP & SIM)

This project provides a **Flask-based REST API** for extracting structured data from Indonesian identity documents, including **ID Cards (KTP)** and **Driving Licenses (SIM)**, using **PaddleOCR**.

It automatically identifies the document type, performs Optical Character Recognition (OCR), intelligently processes the text, and formats the results into a clean, standardized JSON structure.
The pipeline is designed to be **robust against real-world capture issues**, including **minor skew, rotation, and tilt** commonly found in mobile-captured document images.

---

## 🚀 Features

* 🔤 **OCR powered by PaddleOCR (Bahasa Indonesia model)**
* ✅ **Multi-Document Support:** Accurately processes both KTP and SIM cards
* 🔍 **Automatic Document Identification**
* 🧭 **Skew, rotation, and tilt handling** to improve OCR accuracy on misaligned documents
* 🧠 **Smart text processing** using spatial, confidence-based, and regex-driven logic
* 🧹 **Post-processing & normalization** of extracted fields
* 🧪 **OCR diagnostics & debugging support** (bounding boxes, legends, traceability)
* 📦 **Standardized JSON Output** across document types

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
document-ocr/
│
├── app.py                   # Flask API entry point, handles requests & responses
├── document_processor.py    # Orchestrates preprocessing, OCR, and document routing
├── image_preprocessor.py    # Handles image normalization, skew/rotation/tilt correction
├── ktp_extractor.py         # KTP-specific field extraction and normalization logic
├── sim_extractor.py         # SIM-specific field extraction and normalization logic
├── debug_visualizer.py      # OCR debugging tools (bounding boxes, legends, traces)
│
├── uploads/                 # Temporary storage for uploaded document images
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Git ignore rules
```

---

## 🧠 How It Works

1. A client sends a POST request to the API with an image file (**KTP or SIM**) as `multipart/form-data`.
2. The Flask server validates and temporarily stores the image.
3. **Preprocessing logic** normalizes the image and mitigates minor **skew, rotation, or tilt** when detected.
4. The **DocumentProcessor** identifies whether the document is a KTP or SIM.
5. OCR is performed using PaddleOCR.
6. Extractors (`KTPExtractor` / `SIMExtractor`) analyze bounding boxes, spatial alignment, and confidence scores.
7. Fields are cleaned, normalized, and structured.
8. The API returns a standardized JSON response.

---

## 🧭 Orientation & OCR Robustness

To improve accuracy on real-world images, the pipeline includes logic and analysis for:

* Detecting **slight rotation and skew** based on OCR bounding box geometry
* Ensuring consistent line alignment and label–value pairing
* Applying corrections **only when necessary** to avoid disrupting already-correct cases
* Using OCR legends and confidence patterns to guide incremental improvements

Debug artifacts (visual bounding boxes, OCR legends, and extraction traces) are available to support tuning and analysis.

---

## ▶️ Running the Server

```bash
python app.py
```

The API will be available at:

```
http://0.0.0.0:5000
```

---

## 📤 Example API Request

### Endpoint

```
POST /ocr/document
```

### Request Body

* **Type:** `form-data`
* **Key:** `image`
* **Value:** Image file (e.g., `ktp.jpg`)

### Example cURL

```bash
curl -X POST http://localhost:5000/ocr/document \
     -F "image=@/path/to/your_image.jpg"
```
*Replace `/path/to/your_image.jpg` with the actual path to your image file.*

---

### **✅ Example Responses**

#### **KTP Response**

```json
{
    "status": 200,
    "error": false,
    "message": "KTP OCR Processed Successfully",
    "data": {
        "document_type": "KTP",
        "nomor": "3201123456789001",
        "nama": "BUDI SANTOSO",
        "tempat_lahir": "Bandung",
        "tgl_lahir": "01-01-1990",
        "jenis_kelamin": "LAKI-LAKI",
        "agama": "ISLAM",
        "status_perkawinan": "KAWIN",
        "pekerjaan": "KARYAWAN SWASTA",
        "kewarganegaraan": "WNI",
        "alamat": {
            "name": "Jl. Merdeka No. 10",
            "rt_rw": "001/002",
            "kel_desa": "Cihampelas",
            "kecamatan": "Cimahi",
            "kabupaten": "KABUPATEN BANDUNG",
            "provinsi": "JAWA BARAT"
        }
    }
}
```

#### **SIM Response**

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
        "jenis_kelamin": "PRIA",
        "agama": null,
        "status_perkawinan": null,
        "pekerjaan": "PELAJAR/MAHASISWA",
        "kewarganegaraan": null,
        "berlaku_sampai": "06-04-2028",
        "alamat": {
            "name": "JL.H.OYAR NO.24 PEGANGSAAN DUA",
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

You can tune or extend the system by modifying:

* **OCR preprocessing & orientation handling** in `document_processor.py`
* **Field definitions & keywords** in extractor files
* **Regex patterns & confidence thresholds** for specific edge cases
* **Debug visualization & traceability** to analyze OCR behavior

---

## 🧪 Troubleshooting Guide

| Issue                             | Possible Cause         | Recommended Fix                                        |
| --------------------------------- | ---------------------- | ------------------------------------------------------ |
| OCR text misaligned               | Image skew or tilt     | Ensure good lighting; preprocessing handles minor tilt |
| Low confidence fields             | Blurry or angled image | Use higher-resolution input                            |
| OCR returned no results           | Poor image quality     | Retake image with clearer focus                        |
| Could not determine document type | Non-KTP/SIM image      | Upload a valid Indonesian ID                           |
| paddlepaddle not found            | Dependency missing     | Install `paddlepaddle` manually                        |

---

## 🧑‍💻 Author

**Developed by:** *Steffi Soeroredjo*
📧 Email: [steffisoeroredjo5@gmail.com](mailto:steffisoeroredjo5@gmail.com)
🌐 GitHub: [https://github.com/Steffi-Soe](https://github.com/Steffi-Soe)

---