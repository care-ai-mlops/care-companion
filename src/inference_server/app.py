from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
import tritonclient.http as httpclient
import numpy as np
from typing import Dict, Any
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest
import os
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Care Companion Inference Server")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image using NumPy operations"""
    # Resize image
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    # Add batch dimension and ensure correct shape (B, C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))[np.newaxis, ...]
    return img_array

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each set of scores in x"""
    # Subtract max for numerical stability
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")

# Get Triton server URL from environment variable
TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "localhost:8000")

# Initialize Triton client
try:
    triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)
    logger.info(f"Connected to Triton server at {TRITON_SERVER_URL}")
except Exception as e:
    logger.error(f"Failed to connect to Triton server: {e}")
    raise

# Prometheus metrics
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Time spent processing predictions')
PREDICTION_COUNTER = Counter('prediction_total', 'Total number of predictions made')
TRITON_INFERENCE_LATENCY = Histogram('triton_inference_latency_seconds', 'Time spent in Triton server inference')
TRITON_INFERENCE_ERRORS = Counter('triton_inference_errors_total', 'Total number of Triton inference errors')

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if Triton server is ready
        if not triton_client.is_server_ready():
            raise HTTPException(status_code=503, detail="Triton server is not ready")
        return JSONResponse({"status": "healthy", "triton_status": "ready"})
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/home")
async def home():
    """Home endpoint"""
    return JSONResponse({
        "message": "Welcome to Care Companion Inference Server",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "predict_chest": "/predict_chest",
            "predict_wrist": "/predict_wrist",
            "gen_report": "/gen_report"
        }
    })

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.post("/predict_chest")
async def predict_chest(data: Dict[str, Any]):
    """Predict using chest data"""
    PREDICTION_COUNTER.inc()
    start_time = time.time()
    
    try:
        # Extract image data from request
        image_data = data.get("image")
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")

        # Convert base64 image data to PIL Image
        try:
            import base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

        # Convert to RGB if grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Preprocess image using NumPy
        input_data = preprocess_image(image)
        
        # Ensure batch size doesn't exceed model's max_batch_size
        if input_data.shape[0] > 16:
            raise HTTPException(
                status_code=400, 
                detail=f"Batch size {input_data.shape[0]} exceeds maximum allowed batch size of 16"
            )
        
        # Prepare input for Triton
        inputs = [
            httpclient.InferInput("input", input_data.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(input_data)

        # Send inference request to Triton and measure inference time
        triton_start_time = time.time()
        try:
            response = triton_client.infer(
                model_name="chest",
                inputs=inputs
            )
            TRITON_INFERENCE_LATENCY.observe(time.time() - triton_start_time)
        except Exception as e:
            TRITON_INFERENCE_ERRORS.inc()
            raise HTTPException(status_code=503, detail=f"Triton inference failed: {str(e)}")

        # Get prediction results and convert to probabilities using NumPy
        output = response.as_numpy("output")
        probabilities = softmax(output)
        
        # Create class mapping
        class_mapping = {0: "NORMAL", 1: "PNEUMONIA", 2: "TUBERCULOSIS"}
        
        # Convert probabilities to dictionary with class labels
        result = {
            "probabilities": {
                class_mapping[i]: float(prob) 
                for i, prob in enumerate(probabilities[0])
            }
        }
        
        PREDICTION_LATENCY.observe(time.time() - start_time)
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Chest prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_wrist")
async def predict_wrist(data: Dict[str, Any]):
    """Predict using wrist data"""
    PREDICTION_COUNTER.inc()
    start_time = time.time()
    
    try:
        # TODO: Implement wrist prediction logic using Triton
        # This is a placeholder implementation
        result = {"prediction": "wrist_prediction_result"}
        
        PREDICTION_LATENCY.observe(time.time() - start_time)
        return result
    except Exception as e:
        logger.error(f"Wrist prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gen_report")
async def generate_report(data: Dict[str, Any]):
    """Generate a report based on predictions"""
    PREDICTION_COUNTER.inc()
    start_time = time.time()
    
    try:
        # TODO: Implement report generation logic using Triton
        # This is a placeholder implementation
        result = {"report": "generated_report_content"}
        
        PREDICTION_LATENCY.observe(time.time() - start_time)
        return result
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)