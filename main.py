from fastapi import FastAPI,File, UploadFile

import numpy as np
from PIL import Image
from io import BytesIO  
import tensorflow as tf 
from tensorflow import keras


app = FastAPI()

model_path = r"plant_model.keras"

model = keras.models.load_model(model_path)


def read_file(data:bytes)-> np.ndarray:
        # Open image from bytes
    image = Image.open(BytesIO(data)).convert("RGB")

    # Resize to 128x128
    image = image.resize((128, 128))

    # Convert to numpy array
    img_array = np.array(image)

    img_array = img_array / 255.0

    # Add batch dimension → (1, 128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Define class labels (order must match training folder order)
class_labels = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]


@app.post('/predict')
async def predict(file:UploadFile=File(...)):
       img_array=  read_file((await file.read()))
       preds=model.predict(img_array)
       print(preds)
       class_index = int(np.argmax(preds, axis=1)[0])
       confidence = float(np.max(preds))
       return {
      
        "predicted_class": class_labels[class_index],
        "confidence": confidence,

    }