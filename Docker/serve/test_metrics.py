import requests
import time
import random
import base64
from PIL import Image
import io
import numpy as np

def generate_test_image():
    # Create a random test image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def main():
    url = "http://localhost:5000/predict_chest"
    num_requests = 10
    
    print(f"Making {num_requests} requests to generate metrics...")
    
    for i in range(num_requests):
        try:
            # Generate test data with a random image
            data = {
                "image": generate_test_image()
            }
            
            response = requests.post(url, json=data)
            print(f"Request {i+1}: Status {response.status_code}")
            time.sleep(random.uniform(0.1, 0.5))  # Random delay between requests
        except Exception as e:
            print(f"Error on request {i+1}: {e}")
    
    print("Done! Check Grafana for updated metrics.")

if __name__ == "__main__":
    main() 