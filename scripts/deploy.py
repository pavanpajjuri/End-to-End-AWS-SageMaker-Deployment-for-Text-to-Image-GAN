import torch
import torchvision.utils as vutils
import io
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel

s3_bucket = "cv-data-pavanpaj"
model_path = f"s3://{s3_bucket}/saved_models/model.tar.gz"

# Create PyTorch Model for Deployment
pytorch_model = PyTorchModel(
    model_data=model_path,
    role=sagemaker.get_execution_role(),
    framework_version="1.9",
    py_version="py38",
    entry_point="inference.py",  # This script handles requests
    source_dir="scripts"
)

# Deploy the Model as a SageMaker Endpoint
predictor = pytorch_model.deploy(
    instance_type="ml.m5.large",  # Free-tier compatible
    initial_instance_count=1
)

print("SageMaker Endpoint Deployed Successfully!")
