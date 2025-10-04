from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import uvicorn
import numpy as np
import cv2
import tensorflow as tf
from src.data_loader import data_loader
from src.inference import greedy_decode
from config.config import MODEL_PATH, IMG_WIDTH, IMG_HEIGHT


app = FastAPI(title="Captcha OCR API", description="API recongization Captcha with CRNN+CTC", version="1.0")

model = None
characters = None
idx_to_char = None


@app.on_event("startup")
async def startup_event():
    global model, characters, idx_to_char
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    _, _, characters, _, idx_to_char, _, _ = data_loader("captcha_images_v2")
    print("Model loaded succsesfull")


@app.get("/")
async def root():
    return {"message": "OCR captcha API is running"}

@app.get("/health")
async def health_check():   
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True, "message": "API is running healthy"}  

def process_image(file_bytes: bytes):
    file_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  
    img = np.expand_dims(img, axis=0)   
    return img


def predict_captcha(img_tensor):
    preds = model.predict(img_tensor)
    decoded_texts = greedy_decode(preds, characters, idx_to_char)
    return decoded_texts[0]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()
    img_tensor = process_image(file_bytes)
    result = predict_captcha(img_tensor)
    return {"filename": file.filename, "captcha_text": result}

@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        file_bytes = await file.read()
        img_tensor = process_image(file_bytes)
        result = predict_captcha(img_tensor)
        results.append({"filename": file.filename, "captcha_text": result})
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("app.app:app", host="0.0.0.0", port=8000, reload=True)
