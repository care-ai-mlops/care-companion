from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
import tritonclient.http as httpclient
import numpy as np
from typing import Dict, Any, List
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest, Gauge
from prometheus_client.openmetrics.exposition import generate_latest as generate_latest_openmetrics
from prometheus_fastapi_instrumentator import Instrumentator
import os
from PIL import Image
import io
from scipy.stats import entropy, wasserstein_distance, ks_2samp
from collections import deque
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import asyncio
from alibi_detect.cd import MMDDriftOnline
from alibi_detect.cd.pytorch import HiddenOutput, preprocess_drift
import torch
import torch.nn as nn
from torchvision import transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Care Companion Inference Server")

# Initialize Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize drift detection metrics
DRIFT_EVENTS = Counter('drift_events_total', 'Total number of drift events detected')
DRIFT_TEST_STAT = Histogram('drift_test_stat', 'Drift test statistic distribution')
DRIFT_SCORE = Gauge('drift_score', 'Current drift score', ['metric_type'])
DRIFT_THRESHOLD = Gauge('drift_threshold', 'Current drift threshold', ['metric_type'])
DRIFT_WINDOW_SIZE = Gauge('drift_window_size', 'Number of samples in drift detection window')
DRIFT_LAST_UPDATE = Gauge('drift_last_update_timestamp', 'Timestamp of last drift detection update')

# Store historical data for drift detection
class DataStore:
    def __init__(self, max_size=1000):
        self.features = deque(maxlen=max_size)
        self.predictions = deque(maxlen=max_size)
        self.confidences = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.drift_detector = None
        self.drift_executor = ThreadPoolExecutor(max_workers=2)
        self.drift_queue = asyncio.Queue()
        self.drift_task = None
        self.last_drift_update = time.time()

    def initialize_drift_detector(self, model: nn.Module, x_ref: np.ndarray):
        """Initialize the drift detector with reference data"""
        feature_model = HiddenOutput(model, layer=-1)
        preprocess_fn = lambda x: preprocess_drift(x, model=feature_model)
        
        self.drift_detector = MMDDriftOnline(
            x_ref=x_ref,
            ert=300,  # Expected run time between false positives
            window_size=10,  # Number of test samples to use in each MMD check
            backend='pytorch',
            preprocess_fn=preprocess_fn
        )
        
        # Initialize drift metrics
        DRIFT_THRESHOLD.labels(metric_type='mmd').set(self.drift_detector.threshold)
        DRIFT_WINDOW_SIZE.set(self.drift_detector.window_size)

    async def start_drift_monitoring(self):
        """Start the drift monitoring task"""
        if self.drift_task is None:
            self.drift_task = asyncio.create_task(self._process_drift_queue())

    async def _process_drift_queue(self):
        """Process the drift detection queue"""
        while True:
            try:
                features = await self.drift_queue.get()
                if self.drift_detector is not None:
                    # Run drift detection in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.drift_executor,
                        self._detect_drift,
                        features
                    )
                    
                    # Update drift metrics
                    if result['is_drift']:
                        DRIFT_EVENTS.inc()
                    DRIFT_TEST_STAT.observe(result['test_stat'])
                    DRIFT_SCORE.labels(metric_type='mmd').set(result['test_stat'])
                    self.last_drift_update = time.time()
                    DRIFT_LAST_UPDATE.set(self.last_drift_update)
                    
                self.drift_queue.task_done()
            except Exception as e:
                logger.error(f"Error in drift detection: {e}")

    def _detect_drift(self, features: np.ndarray) -> Dict[str, Any]:
        """Detect drift in features using alibi-detect"""
        try:
            result = self.drift_detector.predict(features)
            return result['data']
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return {'is_drift': False, 'test_stat': 0.0}

    def add_prediction(self, features, prediction, confidence):
        with self.lock:
            self.features.append(features)
            self.predictions.append(prediction)
            self.confidences.append(confidence)
            self.timestamps.append(datetime.now())
            
            # Add features to drift detection queue
            if self.drift_detector is not None:
                asyncio.create_task(self.drift_queue.put(features))

    def get_recent_data(self, window_hours):
        with self.lock:
            cutoff = datetime.now() - timedelta(hours=window_hours)
            mask = [t >= cutoff for t in self.timestamps]
            return (
                [f for f, m in zip(self.features, mask) if m],
                [p for p, m in zip(self.predictions, mask) if m],
                [c for c, m in zip(self.confidences, mask) if m]
            )

# Initialize data store
data_store = DataStore()

def calculate_drift_scores(reference_data: List[float], current_data: List[float]) -> Dict[str, float]:
    """Calculate various drift scores between reference and current data using vectorized operations"""
    if not reference_data or not current_data:
        return {"kl_divergence": 0, "wasserstein_distance": 0, "ks_test": 0}
    
    # Convert to numpy arrays for faster computation
    ref_data = np.array(reference_data)
    curr_data = np.array(current_data)
    
    # Calculate histograms in one go
    ref_hist, _ = np.histogram(ref_data, bins=50, density=True)
    curr_hist, _ = np.histogram(curr_data, bins=50, density=True)
    
    # Avoid log(0) and ensure positive values
    ref_hist = np.clip(ref_hist, 1e-10, None)
    curr_hist = np.clip(curr_hist, 1e-10, None)
    
    # Calculate KL divergence using vectorized operations
    kl_div = np.sum(ref_hist * np.log(ref_hist / curr_hist))
    
    # Calculate Wasserstein distance using vectorized operations
    wass_dist = wasserstein_distance(ref_data, curr_data)
    
    # Calculate KS test statistic
    ks_stat, _ = ks_2samp(ref_data, curr_data)
    
    return {
        "kl_divergence": min(kl_div, 1.0),
        "wasserstein_distance": min(wass_dist, 1.0),
        "ks_test": ks_stat
    }

@lru_cache(maxsize=32)
def get_cached_data(window_hours: int) -> tuple:
    """Cache frequently accessed data to avoid repeated calculations"""
    return data_store.get_recent_data(window_hours)

def process_window(window: str, features: List[List[float]], preds: List[int], confs: List[float]) -> None:
    """Process a single time window's data"""
    if not features:
        return
        
    # Get drift detection results from alibi-detect
    if len(features) > 1:
        # Convert features to numpy array
        features_array = np.array(features)
        # Get drift detection result
        drift_result = data_store._detect_drift(features_array)
        
        # Update drift metrics
        if drift_result['is_drift']:
            DRIFT_EVENTS.inc()
        DRIFT_TEST_STAT.observe(drift_result['test_stat'])
        
        # Update data drift score
        DATA_DRIFT_SCORE.labels(metric_name='mmd').set(drift_result['test_stat'])
    
    # Update model accuracy
    if preds and confs:
        accuracy = sum(confs) / len(confs)
        MODEL_ACCURACY.labels(window=window).set(accuracy)

def update_drift_metrics():
    """Update all drift-related metrics using parallel processing"""
    # Get data for different time windows using cached function
    windows_data = {
        "1h": get_cached_data(1),
        "6h": get_cached_data(6),
        "24h": get_cached_data(24)
    }
    
    # Process windows in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for window, (features, preds, confs) in windows_data.items():
            futures.append(executor.submit(process_window, window, features, preds, confs))
        
        # Wait for all futures to complete
        for future in futures:
            future.result()

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
    # Read the HTML file
    with open("static/index.html", "r") as f:
        html_content = f.read()
    
    # Inject the server IP
    server_ip = os.getenv("CHI_FLOATING_IP", "localhost")
    html_content = html_content.replace("</head>", f'<script>window.SERVER_IP = "{server_ip}";</script></head>')
    
    return Response(content=html_content, media_type="text/html")

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

# New metrics for class predictions and confidence
CLASS_PREDICTIONS = Counter('class_predictions_total', 'Total predictions per class', ['class_name'])
PREDICTION_CONFIDENCE = Gauge('prediction_confidence', 'Confidence of the predicted class', ['class_name'])

# Data drift metrics
FEATURE_DISTRIBUTION = Histogram('feature_distribution', 'Distribution of input features', ['feature_name'])
DATA_DRIFT_SCORE = Gauge('data_drift_score', 'Data drift score between training and production data', ['metric_name'])

# Label drift metrics
LABEL_DISTRIBUTION = Counter('label_distribution', 'Distribution of predicted labels', ['class_name'])
LABEL_DRIFT_SCORE = Gauge('label_drift_score', 'Label drift score between training and production data', ['metric_name'])

# Model degradation metrics
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy over time', ['window'])
MODEL_DEGRADATION_SCORE = Gauge('model_degradation_score', 'Model degradation score', ['metric_name'])
CONFIDENCE_DRIFT = Gauge('confidence_drift', 'Drift in model confidence scores', ['class_name'])

# Initialize metrics with default values for all classes
for class_name in ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]:
    CLASS_PREDICTIONS.labels(class_name=class_name)
    PREDICTION_CONFIDENCE.labels(class_name=class_name).set(0)
    LABEL_DISTRIBUTION.labels(class_name=class_name)
    CONFIDENCE_DRIFT.labels(class_name=class_name).set(0)

# Initialize data drift metrics
DATA_DRIFT_SCORE.labels(metric_name='mmd').set(0)

# Initialize model accuracy metrics
for window in ["1h", "6h", "24h"]:
    MODEL_ACCURACY.labels(window=window).set(0)

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
async def predict_chest(data: Dict[str, Any], use_gpu: bool = Query(False)):
    """Predict using chest data with a toggle for GPU or CPU model"""
    PREDICTION_COUNTER.inc()
    start_time = time.time()

    try:
        # Select model based on toggle
        model_name = "chest_gpu" if use_gpu else "chest_openvino"

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

        # Store features for drift detection
        features = input_data.flatten()  # Flatten the image for drift detection

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
                model_name=model_name,
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

        # Get predicted class and confidence
        predicted_class_idx = np.argmax(probabilities[0])
        predicted_class = class_mapping[predicted_class_idx]
        confidence = float(probabilities[0][predicted_class_idx])

        # Store prediction data for drift detection
        data_store.add_prediction(features, predicted_class_idx, confidence)

        # Update drift metrics
        update_drift_metrics()

        # Record metrics
        CLASS_PREDICTIONS.labels(class_name=predicted_class).inc()
        PREDICTION_CONFIDENCE.labels(class_name=predicted_class).set(confidence)

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

@app.on_event("startup")
async def startup_event():
    """Initialize drift detection on startup"""
    try:
        # Load your model here
        model = torch.load("models/chest_model.pth", map_location='cpu')
        model.eval()
        
        # Get reference data path from environment variable with fallback
        reference_data_dir = os.getenv("REFERENCE_DATA_DIR", "/app/reference_data")
        logger.info(f"Using reference data directory: {reference_data_dir}")
        
        if not os.path.exists(reference_data_dir):
            logger.error(f"Reference data directory not found at {reference_data_dir}")
            return
            
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Process each class
        processed_images = []
        expected_classes = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
        missing_classes = []
        
        for class_name in expected_classes:
            class_dir = os.path.join(reference_data_dir, class_name)
            if not os.path.exists(class_dir):
                missing_classes.append(class_name)
                logger.error(f"Class directory not found: {class_dir}")
                continue
                
            # Get list of image files
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                logger.error(f"No images found in class directory: {class_dir}")
                continue
                
            logger.info(f"Processing {len(image_files)} images for class {class_name}")
            
            # Process images
            for img_file in image_files:
                try:
                    img_path = os.path.join(class_dir, img_file)
                    image = Image.open(img_path).convert('RGB')
                    tensor = transform(image)
                    processed_images.append(tensor.numpy())
                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {e}")
                    continue
        
        if missing_classes:
            logger.error(f"Missing reference data for classes: {', '.join(missing_classes)}")
            return
            
        if not processed_images:
            logger.error("No valid images found in reference data")
            return
            
        # Stack all processed images
        reference_data = np.stack(processed_images)
        logger.info(f"Loaded {len(processed_images)} reference images with shape {reference_data.shape}")
        
        # Initialize drift detector
        data_store.initialize_drift_detector(model, reference_data)
        
        # Start drift monitoring
        await data_store.start_drift_monitoring()
        
        logger.info("Drift detection initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize drift detection: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)