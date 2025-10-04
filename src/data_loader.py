import os
import numpy as np
import cv2
from config.config import IMG_HEIGHT, IMG_WIDTH, DATA_DIR, MAX_LABEL_LEN, BATCH_SIZE
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf

def preprocess_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=-1)

def data_loader(data_dir=DATA_DIR):
    all_labels = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".png"):
            label = fname.split(".")[0]
            all_labels.append(label)

    characters = sorted(set("".join(all_labels)))
    char_to_idx = {c: i for i, c in enumerate(characters)}
    idx_to_char = {i: c for i, c in enumerate(characters)}
    max_label_len = MAX_LABEL_LEN
    num_classes = len(characters) + 1 

    def encode_label(label):
        return [char_to_idx[c] for c in label]

    images, labels = [], []
    for fname in os.listdir(data_dir):
        if fname.endswith(".png"):
            label = fname.split(".")[0]
            img_path = os.path.join(data_dir, fname)
            img = preprocess_image(img_path)
            lbl = encode_label(label)
            images.append(img)
            labels.append(lbl)

    X = np.array(images)
    y = np.array(labels)
    y = pad_sequences(labels, maxlen=max_label_len, padding="post", value=len(characters))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    def gen_dataset(X, y):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        ds = ds.map(lambda img, lbl: ({"image": img, "label": lbl}, None))
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds
    
    train_ds = gen_dataset(X_train, y_train)
    val_ds = gen_dataset(X_val, y_val)

    return train_ds, val_ds, characters, char_to_idx, idx_to_char, max_label_len, num_classes
