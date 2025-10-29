// Node.js example for sending an image to the OCR API

const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

// Replace with your OCR server address
const apiUrl = 'http://address.server.ocr:5000/ocr/document';

// Path to the image file you want to send
const pathToImage = 'path/to/your/image.jpg';

// Create a FormData instance and append the image file
const form = new FormData();

// Append the image file to the form data
// First argument is the field name / "Key" expected by the server ('image')
// Second argument is the content of the file (a readable stream in this case)
form.append('image', fs.createReadsStream(pathToImage));

// Send the POST request to the OCR API
axios.post(apiUrl, form,{
    headers: {
        ...form.getHeaders()
    }
})
.then(response => {
    console.log('OCR Response:', response.data);
})
.catch(error => {
    console.error(`API error: ${error.message}`);
    if (error.response) {
        console.error(error.response.data);
    }
})


// Note:
// in form.append('image', ...), the first argument 'image' must match
// the expected field name on the server side for the uploaded file.