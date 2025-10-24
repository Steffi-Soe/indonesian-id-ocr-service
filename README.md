# 🪪 KTP OCR Extractor API

This project provides a **Flask-based REST API** for extracting structured data from **Indonesian ID Cards (KTP)** using **PaddleOCR**.
It performs OCR (Optical Character Recognition), processes text layout intelligently, and formats results into a clean JSON structure.

---

## 🚀 Features

* 🔤 **OCR powered by PaddleOCR (Bahasa Indonesia model)**
* 🧠 **Smart text alignment** using spatial and fuzzy matching
* 🧹 **Post-processing** to clean and normalize field values
* 📦 **JSON API response** with standardized field structure
* 🧾 **Supports multi-line fields** (e.g., address)

---

## 🧰 Requirements

Ensure you have Python **3.8+** installed.

### Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ `paddlepaddle` (backend for PaddleOCR) might need to be installed manually depending on your system:
>
> ```bash
> pip install paddlepaddle==2.6.1
> ```

---

## 📁 Project Structure

```
ktp-ocr/
│
├── app.py                 # Flask web server
├── ktp_extractor.py       # KTP OCR logic & data formatting
├── uploads/               # Directory for uploaded KTP images
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🧠 How It Works

1. You upload or place a KTP image inside the `uploads/` folder.
2. Send a JSON request containing the filename to the API.
3. The `KTPExtractor`:

   * Runs **OCR** using PaddleOCR.
   * Detects fields like `NIK`, `Nama`, `Alamat`, etc.
   * Cleans and normalizes extracted text.
4. The API returns structured JSON output.

---

## ▶️ Running the Server

1. **Start the Flask API:**

   ```bash
   python app.py
   ```

2. **The API will start at:**

   ```
   http://0.0.0.0:5000
   ```

---

## 📤 Example API Request

### **Endpoint**

```
POST /ocr/ktp
```

### **Request Body**

```json
{
    "filename": "ktp_sample.jpg"
}
```

### **Example cURL**

```bash
curl -X POST http://localhost:5000/ocr/ktp \
     -H "Content-Type: application/json" \
     -d '{"filename": "ktp_sample.jpg"}'
```

### **Response**

```json
{
    "status": 200,
    "error": false,
    "message": "Process OCR Successfully",
    "data": {
        "nik": "3201123456789001",
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

---

## 🧩 Key Components

| File                   | Description                                             |
| ---------------------- | ------------------------------------------------------- |
| **`ktp_extractor.py`** | Core OCR extraction and field matching logic.           |
| **`app.py`**           | Flask API server to handle HTTP requests and responses. |
| **`uploads/`**         | Folder where KTP images are stored before processing.   |

---

## ⚙️ Customization

You can modify the following in `ktp_extractor.py`:

* **Field Definitions:** Update `self.canonical_fields` to add or remove fields.
* **OCR Settings:** Adjust `PaddleOCR` parameters (e.g., language or detection sensitivity).
* **Matching Sensitivity:** Tweak fuzzy matching threshold (`score > 85`).

---

## 🧪 Troubleshooting

| Issue                           | Possible Cause         | Fix                                       |
| ------------------------------- | ---------------------- | ----------------------------------------- |
| `Warning: Could not read image` | Wrong filename or path | Check that file exists in `uploads/`.     |
| `OCR returned no results`       | Low-quality image      | Use a clearer or higher-resolution photo. |
| `paddlepaddle not found`        | OCR backend missing    | Install with `pip install paddlepaddle`.  |

---

## 🧑‍💻 Author

**Developed by:** *Steffi Soeroredjo*
📧 Email: [[steffisoeroredjo5@gmail.com](mailto:steffisoeroredjo5@gmail.com)]
🌐 GitHub: [https://github.com/Steffi-Soe](https://github.com/Steffi-Soe)