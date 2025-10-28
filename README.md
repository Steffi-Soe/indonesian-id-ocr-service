#  🪪 🇮🇩 Indonesian Document OCR API (KTP & SIM)

This project provides a **Flask-based REST API** for extracting structured data from Indonesian identity documents, including **ID Cards (KTP)** and **Driving Licenses (SIM)**, using **PaddleOCR**.

It automatically identifies the document type, performs Optical Character Recognition (OCR), processes the text intelligently, and formats the results into a clean, standardized JSON structure.

---

## 🚀 Features

*   🔤 **OCR powered by PaddleOCR (Bahasa Indonesia model)**
*   ✅ **Multi-Document Support:** Accurately processes both KTP and SIM cards.
*   🔍 **Automatic Document Identification:** Intelligently determines the document type before extraction.
*   🧠 **Smart text processing** using spatial and regex-based logic.
*   🧹 **Post-processing** to clean and normalize field values.
*   📦 **Standardized JSON Output:** Provides a consistent JSON structure for different document types.

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
├── app.py                 # Flask web server
├── document_processor.py  # Main processor to identify doc type & route to correct extractor
├── ktp_extractor.py       # KTP-specific OCR and data formatting logic
├── sim_extractor.py       # SIM-specific OCR and data formatting logic
├── uploads/               # Directory for uploaded KTP & SIM images
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🧠 How It Works

1.  You upload a KTP or SIM image to the `uploads/` folder.
2.  Send a JSON request containing the filename to the API.
3.  The **`DocumentProcessor`** identifies whether the image is a KTP or a SIM.
4.  It routes the OCR data to the appropriate extractor (`KTPExtractor` or `SIMExtractor`).
5.  The extractor detects fields, cleans the text, and normalizes the values.
6.  The API returns a structured and standardized JSON output.

---

## ▶️ Running the Server

1.  **Start the Flask API:**

    ```bash
    python app.py
    ```

2.  **The API will start at:**

    ```
    http://0.0.0.0:5000
    ```

---

## 📤 Example API Request

### **Endpoint**

```
POST /ocr/document
```

### **Request Body**

```json
{
    "filename": "your_image_name.jpg"
}
```

### **Example cURL**

```bash
curl -X POST http://localhost:5000/ocr/document \
     -H "Content-Type: application/json" \
     -d '{"filename": "your_image_name.jpg"}'
```

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

| File                        | Description                                                                 |
| --------------------------- | --------------------------------------------------------------------------- |
| **`document_processor.py`** | Identifies the document type and orchestrates the extraction process.       |
| **`ktp_extractor.py`**      | Contains the core logic for KTP data extraction and field matching.         |
| **`sim_extractor.py`**      | Contains the core logic for SIM data extraction and field matching.         |
| **`app.py`**                | The Flask API server that handles HTTP requests and responses.              |
| **`uploads/`**              | Folder where images are stored for processing.                              |

---

## ⚙️ Customization

You can modify the following in the extractor files (`ktp_extractor.py`, `sim_extractor.py`):

*   **Field Definitions:** Update field keywords or add new ones.
*   **OCR Settings:** Adjust `PaddleOCR` parameters in `document_processor.py`.
*   **Parsing Logic:** Refine the regular expressions or fuzzy matching thresholds to improve accuracy for specific edge cases.

---

## 🧪 Troubleshooting

| Issue                               | Possible Cause                      | Fix                                                          |
| ----------------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| `Warning: Could not read image`     | Wrong filename or path              | Check that the file exists in the `uploads/` folder.         |
| `OCR returned no results`           | Low-quality or blurry image         | Use a clearer, higher-resolution photo.                      |
| `Could not determine document type` | Image is not a KTP or SIM           | Ensure the uploaded image is a valid Indonesian KTP or SIM.  |
| `paddlepaddle not found`            | OCR backend is missing              | Install with `pip install paddlepaddle`.                     |

---

## 🧑‍💻 Author

**Developed by:** *Steffi Soeroredjo*
📧 Email: [[steffisoeroredjo5@gmail.com](mailto:steffisoeroredjo5@gmail.com)]
🌐 GitHub: [https://github.com/Steffi-Soe](https://github.com/Steffi-Soe)