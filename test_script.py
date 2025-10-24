import cv2
import os
from paddleocr import PaddleOCR

FOLDER_PATH = 'C:/Users/User/Desktop/ACS/OCR/KTP Extraction/ktp case/'

print("Initializing a fresh PaddleOCR engine for testing...")
ocr_engine = PaddleOCR(use_angle_cls=True, lang='id')
print("Engine initialized.")

try:
    files = os.listdir(FOLDER_PATH)
except FileNotFoundError:
    print(f"\n--- SCRIPT FAILED ---")
    print(f"Error: The folder was not found at the specified path: {FOLDER_PATH}")
    files = []

for filename in files:
    image_path = os.path.join(FOLDER_PATH, filename)

    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"\n--- Processing image: {filename} ---")

        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not read the image: {filename}")
            continue
        
        print("Image loaded successfully. Calling the predict function...")

        try:
            result = ocr_engine.predict(image)
            
            print("\n--- TEST COMPLETE for " + filename + " ---")
            
            if result and isinstance(result, list) and isinstance(result[0], list):
                print("SUCCESS: The 'predict' function returned the correct data structure.")
                print(f"Number of text lines detected: {len(result[0])}")
                print("\nHere are the first 5 detections:")
                for i, line in enumerate(result[0][:5]):
                    text = line[1][0]
                    confidence = line[1][1]
                    print(f"  {i+1}: Text: '{text}', Confidence: {confidence:.4f}")
            else:
                print("FAILURE: The 'predict' function returned an unexpected data structure.")
                print("\n--- Raw Output ---")
                print(result)

        except Exception as e:
            print("\n--- TEST FAILED for " + filename + " ---")
            print(f"An error occurred during the predict call: {e}")
    else:
        print(f"\n--- Skipping non-image file: {filename} ---")

print("\n--- Batch processing complete. ---")