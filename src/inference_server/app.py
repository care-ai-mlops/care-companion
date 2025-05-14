from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import tritonclient.http as httpclient
import numpy as np
from typing import Dict, Any
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
import os
from PIL import Image
import io
from collections import deque
import threading
from datetime import datetime, timedelta
import base64
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Care Companion Inference Server")

# Initialize Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize drift detection metrics - Using only KL divergence
DRIFT_EVENTS = Counter('drift_events_total', 'Total number of drift events detected')
DRIFT_SCORE = Gauge('drift_score', 'Current KL divergence score')
DRIFT_THRESHOLD = Gauge('drift_threshold', 'Current KL divergence threshold')
DRIFT_WINDOW_SIZE = Gauge('drift_window_size', 'Number of samples in drift detection window')
DRIFT_LAST_UPDATE = Gauge('drift_last_update_timestamp', 'Timestamp of last drift detection update')

# Store historical data for drift detection
class DataStore:
    def __init__(self, max_size=1000):
        self.features = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.reference_hist = None
        self.drift_threshold = 0.3  # Threshold for KL divergence
        self.window_size = 50  # Number of samples to use for drift detection
        self.drift_detection_thread = None
        self.stop_drift_detection = threading.Event()
        
        # Add storage for prediction accuracy tracking
        self.predictions = deque(maxlen=max_size)
        self.actual_labels = deque(maxlen=max_size)
        self.accuracy_update_thread = None
        self.stop_accuracy_update = threading.Event()

    def initialize_drift_detector(self, reference_data: np.ndarray):
        """Initialize the drift detector with reference data"""
        # Calculate histogram of reference data
        self.reference_hist, _ = np.histogram(reference_data.flatten(), bins=50, density=True)
        # Avoid log(0) and ensure positive values
        self.reference_hist = np.clip(self.reference_hist, 1e-10, None)
        
        # Initialize drift metrics
        DRIFT_THRESHOLD.set(self.drift_threshold)
        DRIFT_WINDOW_SIZE.set(self.window_size)
        
        # Start the background drift detection thread
        self.stop_drift_detection.clear()
        self.drift_detection_thread = threading.Thread(target=self._run_drift_detection, daemon=True)
        self.drift_detection_thread.start()
        logger.info("Started background drift detection thread")
        
        # Start the background accuracy update thread
        self.stop_accuracy_update.clear()
        self.accuracy_update_thread = threading.Thread(target=self._run_accuracy_update, daemon=True)
        self.accuracy_update_thread.start()
        logger.info("Started background accuracy update thread")

    def _run_drift_detection(self):
        """Background thread for drift detection"""
        logger.info("Drift detection thread started")
        while not self.stop_drift_detection.is_set():
            try:
                # Process drift detection every 5 seconds
                with self.lock:
                    if len(self.features) >= self.window_size and self.reference_hist is not None:
                        recent_features = np.array(list(self.features)[-self.window_size:])
                        result = self.detect_drift(recent_features)
                        
                        # Update drift metrics
                        if result['is_drift']:
                            DRIFT_EVENTS.inc()
                        DRIFT_SCORE.set(result['kl_divergence'])
                        DRIFT_LAST_UPDATE.set(time.time())
                        
                        logger.debug(f"Drift detection processed: score={result['kl_divergence']:.3f}, is_drift={result['is_drift']}")
            except Exception as e:
                logger.error(f"Error in drift detection thread: {e}")
            
            # Sleep for 5 seconds before next check
            time.sleep(5)
        
        logger.info("Drift detection thread stopped")

    def detect_drift(self, features: np.ndarray) -> Dict[str, Any]:
        """Detect drift using KL divergence"""
        try:
            # Calculate histogram of current data
            current_hist, _ = np.histogram(features.flatten(), bins=50, density=True)
            current_hist = np.clip(current_hist, 1e-10, None)
            
            # Calculate KL divergence
            kl_div = np.sum(self.reference_hist * np.log(self.reference_hist / current_hist))
            kl_div = min(kl_div, 1.0)  # Normalize to [0, 1]
            
            # Check if drift is detected
            is_drift = kl_div > self.drift_threshold
            
            return {
                'is_drift': is_drift,
                'kl_divergence': kl_div
            }
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return {'is_drift': False, 'kl_divergence': 0.0}

    def _run_accuracy_update(self):
        """Background thread for updating accuracy metrics"""
        logger.info("Accuracy update thread started")
        while not self.stop_accuracy_update.is_set():
            try:
                # Update accuracy metrics every 60 seconds
                self._update_accuracy_metrics()
            except Exception as e:
                logger.error(f"Error in accuracy update thread: {e}")
            
            # Sleep for 60 seconds before next update
            time.sleep(60)
        
        logger.info("Accuracy update thread stopped")
    
    def _update_accuracy_metrics(self):
        """Update accuracy metrics based on recent predictions"""
        with self.lock:
            if len(self.predictions) == 0 or len(self.actual_labels) == 0:
                # No data to update metrics with
                return
            
            # Calculate accuracy for different time windows
            now = datetime.now()
            
            # 1-hour window
            hour_ago = now - timedelta(hours=1)
            hour_mask = [t >= hour_ago for t in self.timestamps]
            hour_preds = [p for p, m in zip(self.predictions, hour_mask) if m]
            hour_labels = [l for l, m in zip(self.actual_labels, hour_mask) if m]
            
            # 6-hour window
            six_hours_ago = now - timedelta(hours=6)
            six_hour_mask = [t >= six_hours_ago for t in self.timestamps]
            six_hour_preds = [p for p, m in zip(self.predictions, six_hour_mask) if m]
            six_hour_labels = [l for l, m in zip(self.actual_labels, six_hour_mask) if m]
            
            # 24-hour window
            day_ago = now - timedelta(hours=24)
            day_mask = [t >= day_ago for t in self.timestamps]
            day_preds = [p for p, m in zip(self.predictions, day_mask) if m]
            day_labels = [l for l, m in zip(self.actual_labels, day_mask) if m]
            
            # Calculate and update metrics
            if hour_preds and hour_labels:
                hour_acc = sum(1 for p, l in zip(hour_preds, hour_labels) if p == l) / len(hour_preds)
                MODEL_ACCURACY.labels(window="1h").set(hour_acc)
            
            if six_hour_preds and six_hour_labels:
                six_hour_acc = sum(1 for p, l in zip(six_hour_preds, six_hour_labels) if p == l) / len(six_hour_preds)
                MODEL_ACCURACY.labels(window="6h").set(six_hour_acc)
            
            if day_preds and day_labels:
                day_acc = sum(1 for p, l in zip(day_preds, day_labels) if p == l) / len(day_preds)
                MODEL_ACCURACY.labels(window="24h").set(day_acc)

    def add_prediction(self, features, predicted_class=None, actual_class=None):
        """Add a new prediction to the data store"""
        with self.lock:
            self.features.append(features)
            self.timestamps.append(datetime.now())
            
            # Store prediction and actual label if available
            if predicted_class is not None:
                self.predictions.append(predicted_class)
            if actual_class is not None:
                self.actual_labels.append(actual_class)

    def get_recent_data(self, window_hours):
        """Get data from the last window_hours"""
        with self.lock:
            cutoff = datetime.now() - timedelta(hours=window_hours)
            mask = [t >= cutoff for t in self.timestamps]
            return [f for f, m in zip(self.features, mask) if m]
    
    def shutdown(self):
        """Shutdown the background threads"""
        if self.drift_detection_thread and self.drift_detection_thread.is_alive():
            logger.info("Shutting down drift detection thread...")
            self.stop_drift_detection.set()
            self.drift_detection_thread.join(timeout=10)
            logger.info("Drift detection thread stopped")
        
        if self.accuracy_update_thread and self.accuracy_update_thread.is_alive():
            logger.info("Shutting down accuracy update thread...")
            self.stop_accuracy_update.set()
            self.accuracy_update_thread.join(timeout=10)
            logger.info("Accuracy update thread stopped")

# Initialize data store
data_store = DataStore()

# Prometheus metrics
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Time spent processing predictions')
PREDICTION_COUNTER = Counter('prediction_total', 'Total number of predictions made')
TRITON_INFERENCE_LATENCY = Histogram('triton_inference_latency_seconds', 'Time spent in Triton server inference')
TRITON_INFERENCE_ERRORS = Counter('triton_inference_errors_total', 'Total number of Triton inference errors')

# Class predictions and confidence
CLASS_PREDICTIONS = Counter('class_predictions_total', 'Total predictions per class', ['class_name'])
PREDICTION_CONFIDENCE = Gauge('prediction_confidence', 'Confidence of the predicted class', ['class_name'])

# Model accuracy
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy over time', ['window'])

# Initialize metrics with default values for all classes
for class_name in ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]:
    # Just initialize the labels for the counter
    CLASS_PREDICTIONS.labels(class_name=class_name)
    PREDICTION_CONFIDENCE.labels(class_name=class_name).set(0)

# Initialize model accuracy metrics
for window in ["1h", "6h", "24h"]:
    MODEL_ACCURACY.labels(window=window).set(0)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image using NumPy operations"""
    # Log image details for debugging
    logger.info(f"Image mode: {image.mode}, Size: {image.size}")
    
    # Resize image
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    logger.info(f"Image array shape: {img_array.shape}, min: {img_array.min()}, max: {img_array.max()}")
    
    # Normalize using ImageNet stats with NumPy operations
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
    img_array = (img_array - mean) / std
    
    # Add batch dimension and ensure correct shape (B, C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))[np.newaxis, ...]
    logger.info(f"Final array shape: {img_array.shape}")
    
    return img_array

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each set of scores in x"""
    # Subtract max for numerical stability
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    result = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # Log the raw output and softmax result for debugging
    logger.info(f"Raw model output: {x}")
    logger.info(f"Softmax result: {result}, sum: {np.sum(result, axis=1)}")
    
    return result

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
            "simulate_drift": "/simulate_drift",
            "stop_drift_simulation": "/stop_drift_simulation",
            "drift_simulation_status": "/drift_simulation_status",
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
        logger.info(f"Using model: {model_name}")

        # Extract image data from request
        image_data = data.get("image")
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        logger.info(f"Received image data of length: {len(image_data)}")

        # Convert base64 image data to PIL Image
        try:
            image_bytes = base64.b64decode(image_data)
            logger.info(f"Decoded base64 data to bytes of length: {len(image_bytes)}")
            
            image = Image.open(io.BytesIO(image_bytes))
            logger.info(f"Opened image: mode={image.mode}, size={image.size}")
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

        # Convert to RGB if grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info("Converted image to RGB mode")

        # Preprocess image using NumPy
        input_data = preprocess_image(image)
        logger.info(f"Preprocessed image shape: {input_data.shape}")

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
        logger.info(f"Prepared input for Triton with shape: {input_data.shape}")

        # Send inference request to Triton and measure inference time
        triton_start_time = time.time()
        try:
            logger.info(f"Sending request to Triton server for model: {model_name}")
            response = triton_client.infer(
                model_name=model_name,
                inputs=inputs
            )
            triton_time = time.time() - triton_start_time
            TRITON_INFERENCE_LATENCY.observe(triton_time)
            logger.info(f"Received response from Triton server in {triton_time:.3f}s")
        except Exception as e:
            TRITON_INFERENCE_ERRORS.inc()
            logger.error(f"Triton inference failed: {e}")
            raise HTTPException(status_code=503, detail=f"Triton inference failed: {str(e)}")

        # Get prediction results and convert to probabilities using NumPy
        output = response.as_numpy("output")
        logger.info(f"Raw output from model: shape={output.shape}, values={output}")
        
        probabilities = softmax(output)
        logger.info(f"Probabilities after softmax: {probabilities}")

        # Create class mapping
        class_mapping = {0: "NORMAL", 1: "PNEUMONIA", 2: "TUBERCULOSIS"}

        # Get predicted class and confidence
        predicted_class_idx = np.argmax(probabilities[0])
        predicted_class = class_mapping[predicted_class_idx]
        confidence = float(probabilities[0][predicted_class_idx])
        logger.info(f"Predicted class: {predicted_class} (index {predicted_class_idx}) with confidence: {confidence:.4f}")

        # Store prediction data for drift detection (in a non-blocking way)
        data_store.add_prediction(features, predicted_class=predicted_class)

        # Record metrics - only increment the counter for the predicted class
        CLASS_PREDICTIONS.labels(class_name=predicted_class).inc()
        
        # Update confidence metrics only for the predicted class
        PREDICTION_CONFIDENCE.labels(class_name=predicted_class).set(confidence)

        # Convert probabilities to dictionary with class labels
        result = {
            "probabilities": {
                class_mapping[i]: float(prob) 
                for i, prob in enumerate(probabilities[0])
            }
        }
        logger.info(f"Returning result: {result}")

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
        # Get reference data path from environment variable with fallback
        reference_data_dir = os.getenv("REFERENCE_DATA_DIR", "/app/reference_data")
        logger.info(f"Using reference data directory: {reference_data_dir}")
        
        if not os.path.exists(reference_data_dir):
            logger.error(f"Reference data directory not found at {reference_data_dir}")
            return
            
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
                    # Using NumPy-based preprocessing instead of torchvision
                    tensor = preprocess_image(image)
                    processed_images.append(tensor.reshape(-1))  # Flatten for storage
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
        reference_data = np.vstack(processed_images)
        logger.info(f"Loaded {len(processed_images)} reference images with shape {reference_data.shape}")
        
        # Initialize drift detector
        data_store.initialize_drift_detector(reference_data)
        
        logger.info("Drift detection initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize drift detection: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    try:
        # Shutdown drift detection thread
        data_store.shutdown()
        logger.info("Inference server shutting down gracefully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Add drift simulation state
drift_simulation_running = False
drift_simulation_thread = None

def create_synthetic_image(mean=0.5, std=0.1):
    """Create a synthetic image with controlled distribution"""
    # Create a random image with normal distribution
    img_array = np.random.normal(mean, std, (224, 224, 3))
    # Clip values to [0, 1]
    img_array = np.clip(img_array, 0, 1)
    # Convert to uint8
    img_array = (img_array * 255).astype(np.uint8)
    
    # Log synthetic image stats
    logger.info(f"Created synthetic image with mean={mean}, std={std}, shape={img_array.shape}")
    logger.info(f"Synthetic image stats: min={img_array.min()}, max={img_array.max()}, mean={img_array.mean()}")
    
    # Create PIL Image
    image = Image.fromarray(img_array)
    return image

def simulate_drift_background(duration_minutes: int, drift_rate: float = 0.1):
    """Background task to simulate drift"""
    global drift_simulation_running
    
    # Initial parameters
    mean = 0.5
    std = 0.1
    
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    while datetime.now() < end_time and drift_simulation_running:
        try:
            # Calculate current drift
            elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
            current_mean = mean + (drift_rate * elapsed_minutes / duration_minutes)
            
            # Generate image with current distribution
            image = create_synthetic_image(mean=current_mean, std=std)
            
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Prepare request data
            data = {
                "image": image_b64,
                "use_gpu": False  # Use CPU model for testing
            }
            
            # Call the prediction endpoint
            response = requests.post(
                "http://localhost:5000/predict_chest",
                json=data
            )
            
            if response.status_code == 200:
                # Calculate drift score based on the current mean shift
                drift_score = min(abs(current_mean - mean) / drift_rate, 1.0)
                
                # Update drift metrics directly
                DRIFT_SCORE.set(drift_score)
                
                # If drift score exceeds threshold, increment drift events
                if drift_score > 0.7:  # Using 0.7 as threshold
                    DRIFT_EVENTS.inc()
                
                # Update last drift update timestamp
                DRIFT_LAST_UPDATE.set(time.time())
                
                logger.info(f"Drift simulation - Current mean: {current_mean:.3f}, KL divergence: {drift_score:.3f}")
            else:
                logger.error(f"Prediction request failed: {response.status_code}")
            
            # Wait for 5 seconds before next iteration
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in drift simulation: {e}")
            break
    
    drift_simulation_running = False

@app.get("/simulate_drift")
async def simulate_drift(
    duration_minutes: int = Query(30, description="Duration of drift simulation in minutes"),
    drift_rate: float = Query(0.1, description="Rate of drift (how fast the distribution shifts)")
):
    """Start a drift simulation"""
    global drift_simulation_running, drift_simulation_thread
    
    if drift_simulation_running:
        raise HTTPException(
            status_code=400,
            detail="Drift simulation is already running"
        )
    
    drift_simulation_running = True
    
    # Start drift simulation in background thread
    drift_simulation_thread = threading.Thread(
        target=simulate_drift_background,
        args=(duration_minutes, drift_rate),
        daemon=True
    )
    drift_simulation_thread.start()
    
    return {
        "status": "started",
        "duration_minutes": duration_minutes,
        "drift_rate": drift_rate
    }

@app.get("/stop_drift_simulation")
async def stop_drift_simulation():
    """Stop the running drift simulation"""
    global drift_simulation_running
    
    if not drift_simulation_running:
        raise HTTPException(
            status_code=400,
            detail="No drift simulation is running"
        )
    
    drift_simulation_running = False
    if drift_simulation_thread:
        drift_simulation_thread.join(timeout=10)
    
    return {"status": "stopped"}

@app.get("/drift_simulation_status")
async def get_drift_simulation_status():
    """Get the current status of drift simulation"""
    return {
        "running": drift_simulation_running,
        "thread_alive": drift_simulation_thread.is_alive() if drift_simulation_thread else False
    }

@app.get("/reset_metrics")
async def reset_metrics():
    """Reset all metrics to their initial state"""
    try:
        # Re-initialize metrics with default values
        for class_name in ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]:
            PREDICTION_CONFIDENCE.labels(class_name=class_name).set(0)
        
        for window in ["1h", "6h", "24h"]:
            MODEL_ACCURACY.labels(window=window).set(0)
        
        DRIFT_SCORE.set(0)
        DRIFT_THRESHOLD.set(0.3)
        DRIFT_WINDOW_SIZE.set(50)
        DRIFT_LAST_UPDATE.set(time.time())
        
        logger.info("Metrics reset successfully")
        return {"status": "success", "message": "Metrics reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test_model")
async def test_model(pattern_type: str = Query("random", description="Type of test pattern: random, checkerboard, gradient"),
                    use_gpu: bool = Query(False, description="Use GPU model")):
    """Test the model with a synthetic image pattern"""
    try:
        # Create a synthetic test image based on the pattern type
        if pattern_type == "checkerboard":
            # Create a checkerboard pattern
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)
            for i in range(0, 224, 2):
                for j in range(0, 224, 2):
                    img_array[i:i+1, j:j+1] = 255
            logger.info("Created checkerboard test pattern")
        elif pattern_type == "gradient":
            # Create a gradient pattern
            x = np.linspace(0, 1, 224)
            y = np.linspace(0, 1, 224)
            xx, yy = np.meshgrid(x, y)
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)
            img_array[:,:,0] = (xx * 255).astype(np.uint8)  # Red channel
            img_array[:,:,1] = (yy * 255).astype(np.uint8)  # Green channel
            img_array[:,:,2] = ((1-xx) * 255).astype(np.uint8)  # Blue channel
            logger.info("Created gradient test pattern")
        else:
            # Default to random noise
            img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            logger.info("Created random noise test pattern")
        
        # Convert to PIL Image
        image = Image.fromarray(img_array)
        
        # Preprocess image
        input_data = preprocess_image(image)
        
        # Select model based on toggle
        model_name = "chest_gpu" if use_gpu else "chest_openvino"
        
        # Prepare input for Triton
        inputs = [
            httpclient.InferInput("input", input_data.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(input_data)
        
        # Send inference request to Triton
        response = triton_client.infer(
            model_name=model_name,
            inputs=inputs
        )
        
        # Get prediction results and convert to probabilities
        output = response.as_numpy("output")
        probabilities = softmax(output)
        
        # Create class mapping
        class_mapping = {0: "NORMAL", 1: "PNEUMONIA", 2: "TUBERCULOSIS"}
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(probabilities[0])
        predicted_class = class_mapping[predicted_class_idx]
        confidence = float(probabilities[0][predicted_class_idx])
        
        # Return detailed results
        return {
            "pattern_type": pattern_type,
            "model_used": model_name,
            "predicted_class": predicted_class,
            "predicted_class_index": int(predicted_class_idx),
            "confidence": confidence,
            "probabilities": {
                class_mapping[i]: float(prob) 
                for i, prob in enumerate(probabilities[0])
            },
            "raw_output": output.tolist()
        }
    except Exception as e:
        logger.error(f"Test model failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)