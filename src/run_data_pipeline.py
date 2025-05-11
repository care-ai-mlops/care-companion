#!/usr/bin/env python3
from __future__ import annotations
import os
import argparse
from pathlib import Path
from src.data_pipeline import DataPipeline, DataPipelineConfig

def main():
    parser = argparse.ArgumentParser(description="Run the data pipeline for model retraining")
    parser.add_argument("--minio-endpoint", type=str, required=True, help="MinIO endpoint (host:port)")
    parser.add_argument("--minio-access-key", type=str, required=True, help="MinIO access key")
    parser.add_argument("--minio-secret-key", type=str, required=True, help="MinIO secret key")
    parser.add_argument("--minio-bucket", type=str, default="mlflow-artifacts", help="MinIO bucket name")
    parser.add_argument("--mlflow-tracking-uri", type=str, required=True, help="MLflow tracking URI")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory for storing training data")
    parser.add_argument("--batch-size", type=int, default=1000, help="Minimum batch size for retraining")
    parser.add_argument("--retrain-interval", type=int, default=24, help="Retraining interval in hours")
    parser.add_argument("--validation-threshold", type=float, default=0.8, help="Data validation threshold")

    args = parser.parse_args()

    # Create configuration
    config = DataPipelineConfig(
        minio_endpoint=args.minio_endpoint,
        minio_access_key=args.minio_access_key,
        minio_secret_key=args.minio_secret_key,
        minio_bucket=args.minio_bucket,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        retrain_interval_hours=args.retrain_interval,
        validation_threshold=args.validation_threshold
    )

    # Initialize and run pipeline
    pipeline = DataPipeline(config)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main() 