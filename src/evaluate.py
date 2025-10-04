from src.data_loader import data_loader
from src.inference import greedy_decode
import tensorflow as tf
from config.config import DATA_DIR, MODEL_PATH

def sequence_accuracy_metrics(y_true, y_pred):
    correct = sum(1 for gt, pred in zip(y_true, y_pred) if gt == pred)
    return correct / len(y_true)

def character_accuracy_metrics(y_true, y_pred):
    correct, total = 0, 0
    for gt, pred in zip(y_true, y_pred):
        for g, p in zip(gt, pred):
            if g == p:
                correct += 1
            total += 1
    return correct / total

def main():
    _, val_ds, characters, _, idx_to_char, _, _ = data_loader(DATA_DIR)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    images, labels = [], []
    for (batch, _) in val_ds:   
        images.append(batch["image"].numpy())
        labels.append(batch["label"].numpy())

    import numpy as np
    X_val = np.concatenate(images, axis=0)
    y_val = np.concatenate(labels, axis=0)

    y_pred = model.predict(X_val)
    decoded_preds = greedy_decode(y_pred, characters, idx_to_char)


    decoded_labels = []
    for label in y_val:
        decoded_labels.append("".join([idx_to_char[idx] for idx in label if idx < len(idx_to_char)]))
    seq_acc = sequence_accuracy_metrics(decoded_labels, decoded_preds)
    char_acc = character_accuracy_metrics(decoded_labels, decoded_preds)

    print(f"Sequence Accuracy: {seq_acc:.4f}")
    print(f"Character Accuracy: {char_acc:.4f}")

if __name__ == "__main__":
    main()
