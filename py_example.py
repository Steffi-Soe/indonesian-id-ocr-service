# Python example code to send an image file to the OCR API endpoint

import requests
import os

# Path to the image file you want to send
api_url = "http://address.server.ocr:5000/ocr/document"

# Psth to image file to be ran by main system
path_to_image = "/path/to/your/image.jpg"

# Open file with 'read binary' mode ('rb')
with open(path_to_image, 'rb') as file_object:
    # Prepare 'multipart/form-data' payload
    # Create a dictionary with the file to be sent
    # The "Key" should match the expected field name in the server (here, 'image')
    # The "Value" is the file object opened in binary mode
    files_to_send = {
        'image': file_object
    }
    
    try:
        # Send POST request to the API endpoint with the "files"
        response =requests.post(api_url, files=files_to_send)

        # Check API OCR's response status code
        if response.status_code == 200:
            print("OCR Berhasil:")
            print(response.json())
        else:
            print(f"API returned an error: {response.status_code}")
            print(response.json())

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")


# Note: 
# The key, files_to_send = {'image': file_object}: is a dictionary where 'image' is the key that the server expects for the uploaded file.
# `request` library automatically sets the correct 'Content-Type' header for 'multipart/form-data' when using the 'files' parameter 
# and use `image` as the field name for the uploaded file.