import tensorflow as tf
import argparse
import numpy as np
from src.data_loader import preprocess_image, data_loader
from config.config import MODEL_PATH

def greedy_decode(preds, characters, idx_to_char):
    results = []
    for pred in preds:
        pred_idx = np.argmax(pred, axis=-1)
        prev = -1
        s = ""
        for idx in pred_idx:
            if idx != prev and idx < len(characters):
                s += idx_to_char[idx]
            prev = idx
        results.append(s)
    return results

def predict_images(model_path, img_paths, characters, idx_to_char):
    model = tf.keras.models.load_model(model_path, compile=False)

    imgs = preprocess_image(img_paths)
    imgs = np.expand_dims(imgs, axis=0) 
    
    preds = model.predict(imgs)
    decoded_texts = greedy_decode(preds, characters, idx_to_char)

    return decoded_texts[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with trained CRNN + CTC model")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--images", type=str, default='captcha_images_v2/2b827.png')
    args = parser.parse_args()

    _, _, characters, _, idx_to_char, _, _ = data_loader("captcha_images_v2")

    results = predict_images(args.model_path, args.images, characters, idx_to_char)

    print(f"captcha is {results}")
