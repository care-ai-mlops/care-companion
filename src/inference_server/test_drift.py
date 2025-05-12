import numpy as np
import requests
import time
from PIL import Image
import io
import base64
import json
from datetime import datetime, timedelta

def create_synthetic_image(mean=0.5, std=0.1):
    """Create a synthetic image with controlled distribution"""
    # Create a random image with normal distribution
    img_array = np.random.normal(mean, std, (224, 224, 3))
    # Clip values to [0, 1]
    img_array = np.clip(img_array, 0, 1)
    # Convert to uint8
    img_array = (img_array * 255).astype(np.uint8)
    # Create PIL Image
    image = Image.fromarray(img_array)
    return image

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def simulate_drift(base_url="http://localhost:5000", duration_minutes=30):
    """Simulate data drift over time"""
    print(f"Starting drift simulation for {duration_minutes} minutes...")
    
    # Initial parameters
    mean = 0.5
    std = 0.1
    drift_rate = 0.1  # How fast the distribution shifts
    
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    while datetime.now() < end_time:
        # Calculate current drift
        elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
        current_mean = mean + (drift_rate * elapsed_minutes / duration_minutes)
        
        # Generate image with current distribution
        image = create_synthetic_image(mean=current_mean, std=std)
        image_b64 = image_to_base64(image)
        
        # Prepare request data
        data = {
            "image": image_b64,
            "use_gpu": False  # Use CPU model for testing
        }
        
        try:
            # Send request to inference server
            response = requests.post(f"{base_url}/predict_chest", json=data)
            if response.status_code == 200:
                print(f"Request successful - Current mean: {current_mean:.3f}")
            else:
                print(f"Request failed: {response.status_code}")
        except Exception as e:
            print(f"Error sending request: {e}")
        
        # Wait for 5 seconds before next request
        time.sleep(5)

if __name__ == "__main__":
    # You can adjust these parameters
    BASE_URL = "http://localhost:5000/predict_chest"  # Change this to your server URL
    DURATION_MINUTES = 30  # How long to run the simulation
    
    simulate_drift(BASE_URL, DURATION_MINUTES) 