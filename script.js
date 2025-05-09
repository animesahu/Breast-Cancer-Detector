document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');
    const fileNameDisplay = document.getElementById('fileName');
    const analyzeButton = document.getElementById('analyzeButton');
    const loader = document.getElementById('loader');
    const resultText = document.getElementById('resultText');
    const resultDisplay = document.getElementById('resultDisplay');

    let currentFile = null;

    imageUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            if (file.type === "image/jpeg" || file.type === "image/jpg") {
                currentFile = file;
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    imagePreview.style.display = 'block';
                }
                reader.readAsDataURL(file);
                fileNameDisplay.textContent = `Selected: ${file.name}`;
                analyzeButton.disabled = false;
                resetResult();
            } else {
                alert("Please upload a JPG image.");
                imageUpload.value = ""; // Reset file input
                imagePreview.style.display = 'none';
                fileNameDisplay.textContent = '';
                analyzeButton.disabled = true;
                currentFile = null;
                resetResult();
            }
        }
    });

    analyzeButton.addEventListener('click', async () => {
        if (!currentFile) {
            alert("Please select an image first.");
            return;
        }

        loader.style.display = 'block';
        resultText.textContent = 'Analyzing...';
        resultDisplay.className = 'result-default'; // Reset styling
        analyzeButton.disabled = true; // Disable button during analysis

        const formData = new FormData();
        formData.append('image', currentFile); // 'image' is the key your backend expects

        try {
            // --- ACTUAL BACKEND CALL ---
            const response = await fetch('https://breast-cancer-detector-backend.onrender.com/analyze', { // <-- IMPORTANT: URL updated
                method: 'POST',
                body: formData // formData already contains the image with key 'image'
            });

            if (!response.ok) 
                {
                // Try to get more detailed error from backend if possible
                let errorData;
                try {
                    errorData = await response.json();
                } catch (e) {
                    // If backend didn't send JSON error, use status text
                    errorData = { error: response.statusText };
                }
                throw new Error(`HTTP error! status: ${response.status} - ${errorData.error || 'Unknown error'}`);
                }
            const data = await response.json(); // This 'data' will be { "result": 0 } or { "result": 1 }
            // --- END OF ACTUAL BACKEND CALL ---

            // Process the result from 'data'
            if (data.result === 1) 
                {
                resultText.textContent = 'Prediction: Tumor is CANCEROUS';
                resultDisplay.className = 'result-cancerous';
                } 
            else if (data.result === 0) 
                {
                resultText.textContent = 'Prediction: Tumor is NON-CANCEROUS';
                resultDisplay.className = 'result-non-cancerous';
                }
            else {
                // This case might occur if the backend sends an unexpected 'result' value
                // or if 'result' key is missing.
                resultText.textContent = `Error: Unexpected result format from backend: ${JSON.stringify(data)}`;
                resultDisplay.className = 'result-default';
                 }

            } 
            catch (error) 
            {
            console.error('Error analyzing image:', error);
            resultText.textContent = `Error: ${error.message}. Check console for details.`;
            resultDisplay.className = 'result-default';
            } 
            
            finally 
            {
            loader.style.display = 'none';
            analyzeButton.disabled = false;
            }
    });

    
    function resetResult() {
        resultText.textContent = 'Awaiting image analysis...';
        resultDisplay.className = 'result-default';
    }

    // Initial state
    resetResult();
});
