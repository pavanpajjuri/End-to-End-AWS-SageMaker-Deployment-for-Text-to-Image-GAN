# **End-to-End AWS SageMaker Deployment for Text-to-Image GAN**

This document provides a detailed step-by-step explanation of the AWS SageMaker workflow, including setup, model training, deployment, encountered issues, and troubleshooting solutions.

---

## **1. Setting Up AWS SageMaker**

### **Steps:**
1. **Created an AWS Free Tier Account**  
   - Used AWS Free Tier to avoid unnecessary charges.
   - Chose `us-east-2` as the region.

2. **Opened AWS SageMaker Studio**  
   - Launched a SageMaker Notebook Instance with:
     - **Instance Type:** `ml.t3.medium` (Free tier eligible)
     - **Lifecycle Configuration:** Default.

### **Possible Issues and Fixes**
- `ResourceLimitExceeded`: The chosen instance type (`ml.t3.medium`) was unavailable in the selected region.  
  **Fix:** Switched to another instance type like `ml.m5.large`, which is also free-tier eligible for training.
- `AccessDenied`: SageMaker Execution Role lacked permissions to access S3.  
  **Fix:** Attached `AmazonS3FullAccess` and `AmazonSageMakerFullAccess` policies in IAM.

---

## **2. Uploading Data to S3**

### **Steps:**
1. **Created an S3 Bucket** (`cv-data-pavanpaj`).
2. **Uploaded `birds_small.hdf5` dataset to S3** using:
   ```bash
   aws s3 cp birds_small.hdf5 s3://cv-data-pavanpaj/datasets/
   ```

### **Possible Issues and Fixes**
- `AccessDenied: s3:GetObject`: SageMaker lacked permissions to access S3.  
  **Fix:** Updated IAM Role to allow `s3:PutObject` and `s3:GetObject` actions.
- `FileNotFoundError`: The specified dataset path in S3 was incorrect.  
  **Fix:** Listed S3 objects using:
  ```bash
  aws s3 ls s3://cv-data-pavanpaj/
  ```

---

## **3. Training the GAN Model Using SageMaker Notebook**

### **Steps:**
1. **Loaded dataset from S3 into SageMaker Notebook:**
   ```python
   import boto3
   s3 = boto3.client('s3')
   s3.download_file('cv-data-pavanpaj', 'datasets/birds_small.hdf5', 'birds_small.hdf5')
   ```
2. **Implemented the Text-to-Image GAN architecture in PyTorch (`Text_to_Image_GAN_AWS.py`).**
3. **Trained the model using PyTorch on SageMaker Notebook Instance (`ml.t3.medium`).**
4. **Saved the trained model weights (`generator_final.pth`, `discriminator_final.pth`).**
5. **Packaged the model as `model.tar.gz` and uploaded to S3.**
   ```python
   import tarfile
   import os

   # Create tar.gz file
   with tarfile.open("model.tar.gz", "w:gz") as tar:
       tar.add("generator_final.pth", arcname="generator_final.pth")
   ```

### **Possible Issues and Fixes**
- `FileNotFoundError`: Dataset path incorrect while loading.  
  **Fix:** Verified file existence in S3 before running the script.
- `RuntimeError: CUDA out of memory`: The notebook instance lacked GPU memory.  
  **Fix:** Used a smaller batch size (`batch_size=32`) or switched to `ml.g4dn.xlarge` (GPU-enabled).

---

## **4. Deploying the Model as a SageMaker Endpoint**

### **Steps:**
1. **Packaged and uploaded the model (`model.tar.gz`) to S3.**
2. **Created `inference.py` to handle inference requests.**
3. **Wrote `deploy.py` to deploy the model as an endpoint using SageMaker's PyTorchModel API.**
   ```python
   from sagemaker.pytorch import PyTorchModel
   pytorch_model = PyTorchModel(
       model_data="s3://cv-data-pavanpaj/saved_models/model.tar.gz",
       role="AmazonSageMaker-ExecutionRole",
       framework_version="1.9",
       py_version="py38",
       entry_point="inference.py",
   )

   predictor = pytorch_model.deploy(instance_type="ml.m5.large", initial_instance_count=1)
   ```
4. **Ran `deploy.py` to deploy the model as an endpoint.**
   ```bash
   python scripts/deploy.py
   ```

### **Possible Issues and Fixes**
- `FileNotFoundError: inference.py`: SageMaker could not find `inference.py` during deployment.  
  **Fix:** Ensured `inference.py` was inside `scripts/` and specified `source_dir="scripts"` in `deploy.py`.
- `ModelError: Invocation Timed Out`: The endpoint took too long to respond.  
  **Fix:**  
  - Optimized `inference.py` to remove unnecessary computations.
  - Increased timeout settings in `deploy.py`:
    ```python
    predictor = pytorch_model.deploy(container_startup_health_check_timeout=600)
    ```
  - Switched to a more powerful instance (`ml.g4dn.xlarge`).

---

## **5. Testing the Deployed Endpoint**

### **Steps:**
1. **Sent an inference request using a JSON payload:**
   ```python
   import json
   import numpy as np
   from sagemaker.predictor import Predictor

   text_embedding = np.random.rand(1024).tolist()
   payload = json.dumps({"text_embedding": text_embedding})

   endpoint_name = "pytorch-inference-2025-02-28-23-37-52-322"
   predictor = Predictor(endpoint_name)

   response = predictor.predict(payload)

   with open("generated_image.png", "wb") as f:
       f.write(response)
   print("Generated Image Saved!")
   ```

### **Possible Issues and Fixes**
- `ModelError: Received server error`:  
  **Fix:** Checked CloudWatch logs for debugging:
  ```bash
  aws logs describe-log-groups --log-group-name /aws/sagemaker/Endpoints/pytorch-inference-2025-02-28-23-37-52-322
  ```
- `KeyError: 'text_embedding'`:  
  **Fix:** Verified the request payload format to ensure it contained the correct key.

---

## **6. Cleaning Up AWS Resources**

### **Steps:**
1. **Deleted the SageMaker Endpoint:**
   ```python
   from sagemaker import Session
   session = Session()
   session.delete_endpoint("pytorch-inference-2025-02-28-23-37-52-322")
   ```
2. **Deleted the Endpoint Configuration:**
   ```python
   sagemaker_client.delete_endpoint_config(EndpointConfigName="pytorch-inference-2025-02-28-23-37-52-322")
   ```
3. **Deleted the Model from SageMaker:**
   ```python
   sagemaker_client.delete_model(ModelName="pytorch-inference-2025-02-28-23-37-52-322")
   ```
4. **Deleted S3 Model Artifacts:**
   ```python
   s3 = boto3.client("s3")
   s3.delete_object(Bucket="cv-data-pavanpaj", Key="saved_models/model.tar.gz")
   ```
5. **Stopped and Deleted SageMaker Notebook Instance:**
   ```python
   sm = boto3.client("sagemaker")
   sm.stop_notebook_instance(NotebookInstanceName="your-notebook-name")
   sm.delete_notebook_instance(NotebookInstanceName="your-notebook-name")
   ```

### **Possible Issues and Fixes**
- `Endpoint not found`: The endpoint name was incorrect.  
  **Fix:** Listed endpoints before deleting:
  ```python
  response = sagemaker_client.list_endpoints()
  print(response)
  ```

---

## **Summary of the End-to-End Process**
1. **Set up AWS SageMaker and configured IAM permissions.**
2. **Uploaded the dataset to S3.**
3. **Trained the Text-to-Image GAN model using PyTorch on SageMaker Notebook.**
4. **Packaged and uploaded the trained model to S3 as `model.tar.gz`.**
5. **Deployed the model as a SageMaker Endpoint.**
6. **Sent an inference request and generated an image.**
7. **Troubleshot errors including model loading, timeout, and S3 access issues.**
8. **Deleted all AWS resources to avoid unnecessary charges.**

---

## **Future Enhancements**
- Automate model retraining using **SageMaker Pipelines**.
- Deploy the model as a **real-time API using AWS Lambda and API Gateway**.
- Set up **AutoScaling** to optimize costs.

This document serves as a complete reference for the AWS SageMaker workflow.
