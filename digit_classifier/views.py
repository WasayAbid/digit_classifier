from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import io
from scipy.ndimage import median_filter  # Import for median filtering
import os

# Define model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mnist_cnn_model.keras")

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Function to render the homepage
def home(request):
    return render(request, "index.html")

# API root for DRF
@api_view(["GET"])
def api_root(request):
    return JsonResponse({"predict_digit": request.build_absolute_uri("/predict/")})

# Function to handle digit classification
@api_view(["POST"])
def classify_digit(request):
    if "image" not in request.FILES:
        return JsonResponse({"error": "No image uploaded"}, status=400)

    image_file = request.FILES["image"]
    image = Image.open(io.BytesIO(image_file.read()))

    # Convert image to grayscale
    image = image.convert("L")

    # Resize to 28x28 pixels
    image = image.resize((28, 28))

    # Convert image to numpy array and normalize
    image_array = np.array(image).astype(np.float32) / 255.0  # Ensure float32

    # Apply median filter for noise reduction
    image_array = median_filter(image_array, size=3)  # Adjust size as needed

    # Check if the image needs inversion based on mean pixel intensity
    mean_intensity = np.mean(image_array)

    # If mean intensity is closer to 1 (white), and we expect black digits
    if mean_intensity > 0.5:  # Adjust threshold as needed
        image_array = 1 - image_array  # Invert the pixel values

    # Reshape to fit model input (1, 28, 28, 1)
    image_array = image_array.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return JsonResponse({"predicted_digit": int(predicted_digit), "confidence": confidence})
