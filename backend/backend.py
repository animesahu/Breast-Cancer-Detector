import cv2
import numpy as np
from skimage import measure, filters, morphology
from skimage.feature import graycomatrix, graycoprops
import joblib
import io # For handling image bytes

from flask import Flask, request, jsonify
from flask_cors import CORS # For handling Cross-Origin Resource Sharing

# ==================
# Initialize Flask App
# ==================
app = Flask(__name__)
CORS(app) # This enables CORS for all routes, allowing your frontend to call this backend

# ==================
# Feature Extraction Functions (Your existing functions)
# ==================
def boxcount(Z, k):
    S = np.add.reduceat(
         np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                           np.arange(0, Z.shape[1], k), axis=1)
    return np.sum(S > 0)

def fractal_dimension(Z):
    assert Z.ndim == 2, "Input image must be two dimensional"
    Z = (Z < 128)
    p = min(Z.shape)
    n = 2 ** np.floor(np.log2(p))
    sizes = 2 ** np.arange(np.floor(np.log2(n)), 1, -1)
    counts = []
    for size in sizes:
         counts.append(boxcount(Z, int(size)))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def compute_aggregate(feature_list):
    arr = np.array(feature_list)
    if len(arr) == 0: # Handle case with no features (e.g., no cells found)
        return 0.0, 0.0, 0.0 # Return default values
    mean_val = np.mean(arr)
    se_val = np.std(arr) / (np.sqrt(len(arr)) + 1e-9) # avoid division by zero if len(arr) is 1
    worst_val = np.max(arr)
    return mean_val, se_val, worst_val

# ==================
# Model Loading (Load it once when the server starts)
# ==================
try:
    loaded_model = joblib.load('log_model.pkl') # Make sure 'log_model.pkl' is in the same directory or provide full path
    print("Model loaded successfully.")
except FileNotFoundError:
    print("ERROR: log_model.pkl not found. Make sure the model file is in the correct location.")
    loaded_model = None # Or handle this error more gracefully

# ==================
# Flask Route for Image Analysis
# ==================
@app.route('/analyze', methods=['POST']) # This is the endpoint your frontend will call
def analyze_image_route():
    if not loaded_model:
        return jsonify({'error': 'Model not loaded, cannot process request.'}), 500

    # 1. Receive the image from the frontend request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided in the request.'}), 400
    
    file = request.files['image'] # 'image' is the name we used in FormData on the frontend

    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    if file:
        try:
            # Read image file bytes
            img_bytes = file.read()
            # Convert bytes to numpy array for OpenCV
            nparr = np.frombuffer(img_bytes, np.uint8)
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if image is None:
                return jsonify({'error': 'Could not decode image. Is it a valid JPG?'}), 400

            # 2. Your existing image processing and feature extraction logic
            #    (Copied and adapted from your script)
            
            # STEP 1: Preprocess Image & Segment Nuclei
            blurred = cv2.GaussianBlur(image, (5,5), 0)
            thresh_val = filters.threshold_otsu(blurred)
            binary = blurred > thresh_val
            binary = morphology.remove_small_objects(binary, min_size=30)
            binary = morphology.remove_small_holes(binary, area_threshold=30)
            labels = measure.label(binary)
            props = measure.regionprops(labels, intensity_image=image)

            # STEP 2: Extract Per-Cell Features
            radius_list, texture_list, perimeter_list, area_list = [], [], [], []
            smoothness_list, compactness_list, concavity_list = [], [], []
            concave_points_list, symmetry_list, fractal_dimension_list = [], [], []

            for prop in props:
                if prop.area < 50: continue
                radius_list.append(prop.equivalent_diameter / 2.0)
                minr, minc, maxr, maxc = prop.bbox
                region_intensity = image[minr:maxr, minc:maxc]
                if region_intensity.size == 0 or region_intensity.shape[0] < 1 or region_intensity.shape[1] < 1:
                    texture_list.append(0) # Default if region is empty or too small
                else:
                    glcm = graycomatrix(region_intensity, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
                    texture_list.append(graycoprops(glcm, prop='contrast')[0,0])
                perimeter_list.append(prop.perimeter)
                area_list.append(prop.area)
                coords = prop.coords
                centroid = np.array(prop.centroid)
                distances = np.sqrt(np.sum((coords - centroid)**2, axis=1))
                smoothness_list.append(np.std(distances) / (np.mean(distances) + 1e-5))
                compactness_list.append((prop.perimeter ** 2) / (prop.area + 1e-5))
                concavity_list.append(prop.convex_area - prop.area)
                concave_points_list.append(prop.convex_area - prop.area) # Same as concavity
                symmetry_list.append(prop.major_axis_length / (prop.minor_axis_length + 1e-5))
                fractal_dimension_list.append(fractal_dimension(prop.image))

            # STEP 3: Compute Aggregate Features
            features = {}
            if not radius_list: # No cells found or processed
                 # Create a default feature vector of zeros if no cells are found
                 # This ensures the model receives an input of the correct shape.
                 # The number of zeros should match the expected number of features (30 in this case).
                default_feature_values = np.zeros(30)
                base_feature_names = [
                    'radius', 'texture', 'perimeter', 'area', 'smoothness',
                    'compactness', 'concavity', 'concave points', 'symmetry',
                    'fractal_dimension'
                ]
                suffixes_ordered = ['_mean', '_se', '_worst']
                idx = 0
                for base_name in base_feature_names:
                    for suffix in suffixes_ordered:
                        features[f"{base_name}{suffix}"] = default_feature_values[idx]
                        idx +=1
                print("No valid cells found; using default zero features.")

            else:
                features['radius_mean'], features['radius_se'], features['radius_worst'] = compute_aggregate(radius_list)
                features['texture_mean'], features['texture_se'], features['texture_worst'] = compute_aggregate(texture_list)
                features['perimeter_mean'], features['perimeter_se'], features['perimeter_worst'] = compute_aggregate(perimeter_list)
                features['area_mean'], features['area_se'], features['area_worst'] = compute_aggregate(area_list)
                features['smoothness_mean'], features['smoothness_se'], features['smoothness_worst'] = compute_aggregate(smoothness_list)
                features['compactness_mean'], features['compactness_se'], features['compactness_worst'] = compute_aggregate(compactness_list)
                features['concavity_mean'], features['concavity_se'], features['concavity_worst'] = compute_aggregate(concavity_list)
                features['concave points_mean'], features['concave points_se'], features['concave points_worst'] = compute_aggregate(concave_points_list)
                features['symmetry_mean'], features['symmetry_se'], features['symmetry_worst'] = compute_aggregate(symmetry_list)
                features['fractal_dimension_mean'], features['fractal_dimension_se'], features['fractal_dimension_worst'] = compute_aggregate(fractal_dimension_list)

            # Order features consistently for the model
            # This order MUST match the order of features your model was trained on.
            ordered_feature_keys = [
                'radius_mean', 'radius_se', 'radius_worst',
                'texture_mean', 'texture_se', 'texture_worst',
                'perimeter_mean', 'perimeter_se', 'perimeter_worst',
                'area_mean', 'area_se', 'area_worst',
                'smoothness_mean', 'smoothness_se', 'smoothness_worst',
                'compactness_mean', 'compactness_se', 'compactness_worst',
                'concavity_mean', 'concavity_se', 'concavity_worst',
                'concave points_mean', 'concave points_se', 'concave points_worst',
                'symmetry_mean', 'symmetry_se', 'symmetry_worst',
                'fractal_dimension_mean', 'fractal_dimension_se', 'fractal_dimension_worst'
            ]
            
            features_for_model = [features[key] for key in ordered_feature_keys]
            features_np = np.array(features_for_model).reshape(1, -1)

            # 3. Use the loaded model for prediction
            predictions = loaded_model.predict(features_np)
            prediction_result = int(predictions[0]) # Convert to standard Python int

            # 4. Send the result back to the frontend as JSON
            return jsonify({'result': prediction_result})

        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500

    return jsonify({'error': 'Unknown error occurred.'}), 500


# ==================
# Run the Flask App
# ==================
if __name__ == '__main__':
    # Port 5000 is a common default for Flask development
    # debug=True is helpful during development for error messages, but turn off for production
    app.run(debug=True, port=5000)