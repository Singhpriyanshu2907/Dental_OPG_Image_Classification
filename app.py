import io
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms
import sys
import os
from config.model_trainer_config import (
    CLASS_LABELS, 
    IMG_SIZE, 
    NUM_CLASSES, 
    OVERALL_BEST_MODEL_PATH, 
    DEVICE
)

from src.model_trainer import ModelZoo


# --- API Response Model ---
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float

# --- Model Loading ---
try:
    best_model_path = os.path.join(OVERALL_BEST_MODEL_PATH, "best_overall_model.pth")
    
    # We need the model name to initialize it correctly.
    # You might need to change this if you're not using resnet18.
    model_name = "resnet18"
    
    # The following line uses the imported variables directly
    model = ModelZoo.get_model(model_name, NUM_CLASSES)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()
    
    # Set the device
    device = torch.device(DEVICE)
    model.to(device)

except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)


# --- Image Transformations ---
# These should match the transformations used during your model's training
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE), # Now uses the variable directly
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# --- FastAPI Application ---
app = FastAPI()

def predict_class(image: Image.Image) -> tuple[str, float]:
    """
    Performs inference on a single image and returns the predicted class and confidence.
    """
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
    confidence, predicted_index = torch.max(probabilities, 1)
    
    # Now uses the imported CLASS_LABELS list
    predicted_label = CLASS_LABELS[predicted_index.item()]
    
    return predicted_label, confidence.item()


@app.get("/")
def read_root():
    return {"message": "Welcome to the Dental OPG Image Classification API"}


@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Endpoint for making predictions on an uploaded dental OPG image.
    """
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        predicted_label, confidence = predict_class(image)

        return PredictionResponse(
            predicted_class=predicted_label,
            confidence=confidence
        )
    
    except Exception as e:
        return {"error": str(e)}