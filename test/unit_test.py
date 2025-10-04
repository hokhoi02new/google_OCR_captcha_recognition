import numpy as np
import cv2
from app.app import process_image 
from src.inference import greedy_decode

IMG_HEIGHT, IMG_WIDTH = 50, 200

def test_process_image():
    # Tạo ảnh giả bằng OpenCV (ảnh trắng)
    dummy_img = np.ones((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8) * 255
    _, buf = cv2.imencode(".png", dummy_img)   # encode thành bytes
    file_bytes = buf.tobytes()

    # Gọi hàm process_image
    tensor = process_image(file_bytes)

    # Kiểm tra shape
    assert tensor.shape == (1, IMG_HEIGHT, IMG_WIDTH, 1)
    # Pixel đã được normalize về [0,1]
    assert tensor.max() <= 1.0 and tensor.min() >= 0.0


def test_greedy_decode_simple():
    # Giả lập output của model: (batch=1, timesteps=5, classes=4)
    # classes = 0:'A', 1:'B', 2:'C', 3:'blank'
    characters = ["A", "B", "C"]
    idx_to_char = {0:"A", 1:"B", 2:"C"}

    preds = np.array([[
        [0.9, 0.05, 0.05, 0.0],  # A
        [0.8, 0.1, 0.1, 0.0],    # A
        [0.1, 0.8, 0.1, 0.0],    # B
        [0.05, 0.05, 0.9, 0.0],  # C
        [0.9, 0.05, 0.05, 0.0],  # A
    ]])

    result = greedy_decode(preds, characters, idx_to_char)

    # Vì greedy chọn argmax -> chuỗi "ABCA"
    assert result[0] == "ABCA"
