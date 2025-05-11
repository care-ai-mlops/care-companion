from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import time
import logging
from typing import Optional, List, Dict, Any
import boto3
from botocore.exceptions import ClientError
import pandas as pd
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataPipelineConfig:
    """Configuration for the data pipeline."""
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    mlflow_tracking_uri: str
    data_dir: Path
    batch_size: int = 1000
    retrain_interval_hours: int = 240
    validation_threshold: float = 0.8

class DataPipeline:
    """Handles data pipeline operations including batch retraining, data extraction, and validation."""
    
    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self.s3_client = boto3.client(
            's3',
            endpoint_url=f'http://{config.minio_endpoint}',
            aws_access_key_id=config.minio_access_key,
            aws_secret_access_key=config.minio_secret_key
        )
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        self.mlflow_client = MlflowClient()

    def extract_new_data(self, start_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Extract new data from MinIO that was uploaded after start_time.
        If start_time is None, extracts all data.
        """
        try:
            if start_time is None:
                start_time = datetime.now() - timedelta(hours=self.config.retrain_interval_hours)

            # List objects in the bucket
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.minio_bucket,
                Prefix='inference_data/'
            )

            new_data = []
            for obj in response.get('Contents', []):
                if obj['LastModified'] > start_time:
                    # Download and process the file
                    response = self.s3_client.get_object(
                        Bucket=self.config.minio_bucket,
                        Key=obj['Key']
                    )
                    # Process the data based on your format
                    # This is a placeholder - adjust based on your data format
                    data = pd.read_csv(response['Body'])
                    new_data.append(data)

            if not new_data:
                logger.info("No new data found")
                return pd.DataFrame()

            return pd.concat(new_data, ignore_index=True)

        except ClientError as e:
            logger.error(f"Error accessing MinIO: {e}")
            raise

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the extracted data.
        Returns True if data passes validation, False otherwise.
        """
        if data.empty:
            logger.warning("Empty dataset")
            return False

        # Add your validation rules here
        # Example validations:
        required_columns = ['image_path', 'label', 'timestamp']
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns. Required: {required_columns}")
            return False

        # Check for missing values
        if data.isnull().any().any():
            logger.warning("Dataset contains missing values")
            return False

        # Check data quality metrics
        # Add your specific validation rules here
        return True

    def prepare_training_data(self, new_data: pd.DataFrame) -> None:
        """
        Prepare the data for training by moving it to the appropriate location
        and updating the dataset splits.
        """
        if new_data.empty:
            logger.info("No new data to prepare")
            return

        # Create timestamp-based directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = self.config.data_dir / f"batch_{timestamp}"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save the new data
        new_data.to_csv(data_dir / "new_data.csv", index=False)
        logger.info(f"Saved new data to {data_dir}")

    def check_retraining_trigger(self) -> bool:
        """
        Check if retraining should be triggered based on:
        1. Time since last training
        2. Amount of new data
        3. Model performance metrics
        """
        try:
            # Get the latest model version
            latest_model = self.mlflow_client.get_latest_versions("chest-classifier-model")[0]
            last_training_time = datetime.fromtimestamp(latest_model.creation_timestamp / 1000)
            
            # Check time-based trigger
            time_since_training = datetime.now() - last_training_time
            if time_since_training > timedelta(hours=self.config.retrain_interval_hours):
                logger.info("Retraining triggered: Time threshold exceeded")
                return True

            # Check data volume trigger
            new_data = self.extract_new_data(last_training_time)
            if len(new_data) >= self.config.batch_size:
                logger.info("Retraining triggered: New data threshold exceeded")
                return True

            # Check model performance trigger
            # Add your model performance checks here
            # Example: if accuracy drops below threshold

            return False

        except Exception as e:
            logger.error(f"Error checking retraining trigger: {e}")
            return False

    def run_pipeline(self) -> None:
        """
        Run the complete data pipeline:
        1. Check if retraining is needed
        2. Extract new data if needed
        3. Validate the data
        4. Prepare data for training
        """
        try:
            if not self.check_retraining_trigger():
                logger.info("No retraining needed at this time")
                return

            new_data = self.extract_new_data()
            if not self.validate_data(new_data):
                logger.error("Data validation failed")
                return

            self.prepare_training_data(new_data)
            logger.info("Data pipeline completed successfully")

        except Exception as e:
            logger.error(f"Error in data pipeline: {e}")
            raise 