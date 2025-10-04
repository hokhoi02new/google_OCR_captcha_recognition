import argparse
import tensorflow as tf
from src.data_loader import data_loader
from src.model import build_crnn_ctc
from config.config import DATA_DIR, IMG_HEIGHT, IMG_WIDTH, MODEL_PATH, EPOCHS, BATCH_SIZE, LEARNING_RATE, RNN_UNITS


def train(args):
    train_ds, val_ds, _, _, _, max_label_len, num_classes = data_loader(DATA_DIR)

    train_model, pred_model = build_crnn_ctc(IMG_HEIGHT, IMG_WIDTH, num_classes, 128, max_label_len)

    train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr))

    train_model.fit(
    x=train_ds,
    validation_data=val_ds,
    batch_size=32,
    epochs=100)

    pred_model.save(MODEL_PATH) 
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRNN + CTC for CAPTCHA OCR")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--rnn_units", type=int, default=RNN_UNITS, help="Number of units in LSTM")
    args = parser.parse_args()

    train(args)
