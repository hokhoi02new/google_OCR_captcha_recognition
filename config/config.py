# config.py
import os

# Dataset
DATA_DIR = "captcha_images_v2"

# Image parameters
IMG_HEIGHT = 50
IMG_WIDTH = 200

# Training parameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
RNN_UNITS = 128
MAX_LABEL_LEN = 5

# Paths
MODEL_DIR = "models"
MODEL_PATH = 'models/ocr_crnn_ctc.h5'
