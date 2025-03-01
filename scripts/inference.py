import torch
import os
import io
import json
import time
import logging
import numpy as np
from torchvision import transforms
from PIL import Image
from Text_to_Image_GAN_AWS import G  # ✅ Import Generator model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Model Function (SageMaker will call this)
def model_fn(model_dir):
    """Loads the trained model when the endpoint is started."""
    logger.info("Loading model from: " + model_dir)

    model_path = os.path.join(model_dir, "generator_final.pth")  # ✅ Fix model path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Generator Model
    generator = G(embed_dim=1024, embed_out_dim=128).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    logger.info("Model loaded successfully.")
    return generator

# Transform Incoming Request and Generate Image
def transform_fn(model, request_body, content_type, accept_type):
    """Handles inference requests."""
    logger.info("Received request for inference...")

    try:
        start_time = time.time()

        # Convert JSON to tensor
        request = json.loads(request_body)
        if "text_embedding" not in request:
            raise ValueError("Invalid request format: 'text_embedding' key missing")

        text_embedding = torch.tensor(request["text_embedding"], dtype=torch.float32).unsqueeze(0)

        logger.info(f"Text Embedding Shape: {text_embedding.shape}")

        # Generate Image
        noise = torch.randn(1, 100, 1, 1)
        with torch.no_grad():
            generated_image = model(noise, text_embedding)

        # Convert tensor to image
        image = transforms.ToPILImage()(generated_image.squeeze(0).cpu())

        # Save image to buffer
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        latency = time.time() - start_time
        logger.info(f"Inference completed in {latency:.2f} seconds.")

        return buffer.getvalue(), accept_type

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise
