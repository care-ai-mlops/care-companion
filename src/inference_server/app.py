from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import tritonclient.http as httpclient
import numpy as np
from typing import Dict, Any
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Care Companion Inference Server")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if Triton server is ready
        if not triton_client.is_server_ready():
            raise HTTPException(status_code=503, detail="Triton server is not ready")
        return {"status": "healthy", "triton_status": "ready"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/home")
async def home():
    """Home endpoint"""
    return {
        "message": "Welcome to Care Companion Inference Server",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "predict_chest": "/predict_chest",
            "predict_wrist": "/predict_wrist",
            "gen_report": "/gen_report"
        }
    }

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

        # Convert image data to numpy array
        input_data = np.array(image_data, dtype=np.float32)
        
        # Prepare input for Triton
        inputs = [
            httpclient.InferInput("input", input_data.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(input_data)

        # Send inference request to Triton
        response = triton_client.infer(
            model_name="chest_model",  # Replace with your actual model name
            inputs=inputs
        )

        # Get prediction results
        output = response.as_numpy("output")  # Replace "output" with your model's output name
        
        PREDICTION_LATENCY.observe(time.time() - start_time)
        return {"prediction": output.tolist()}
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