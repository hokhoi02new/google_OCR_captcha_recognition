## 🚀 Introduction
This project builds an OCR system for Captcha recognition using deep learning approach **CRNN (CNN + BiLSTM) + CTC Loss**, provide RESTful API using FastAPI and a web UI using streamlit, both service deploy on cloud platform (Render). The model achieved 94.3% SeqAcc (Sequence Accuracy) metrics on the evaluation dataset.

Key features:
- 🧠 **Deep Learning Model**: Combine Deep-CNN for feature extraction, RNN (BiLSTM) for sequence modeling, and CTC for alignment-free text recognition.  
- ⚡ **Backend API (FastAPI)**: A production-ready API for real-time captcha recognition.  
- 🎨 **Frontend UI (Streamlit)**: A user interface to upload captcha images and view predictions.  
- 📦 **Dockerized Deployment**: To deployment on cloud services (Render)
  
---
## 🌐 Demo
<img width="900" height="900" alt="Screenshot 2025-10-01 212046" src="https://github.com/user-attachments/assets/166d7bb8-435f-4019-8a14-42fd71a5248a" />
---

## 📂 Project Structure
```
.
├── app/                 
│   ├── app.py            # API
│   └── app_ui.py         # UI
├── captcha_images_v2/    # dataset
├── config/               # configuration files
│   └── config.py
├── models/               # saved model
│   └──ocr_crnn_ctc.h5    
├── src/                  # main scripts
│   ├── data_loader.py    
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── evaluate.py
├── test/                 # Unit tests & integration tests
│   ├── test_api.py
│   └── unit_test.py
├── Docker-UI 
├── Docker-API
├── requirements-api.txt
├── requirements-ui.txtas
└── README.md
```

⚠️ Note: The `captcha_images_v2/` folders are empty in this repository because the dataset are too large for GitHub.  
You can download them from the following links: https://drive.google.com/file/d/1bhIKAr4Z-g123u96lxNnY06iQyRVnkNG/view?usp=drive_link

---
## ⚙️ Usage

#### Clone the repository

```bash
git clone https://github.com/hokhoi02new/google_OCR_captcha_recognition.git
cd google_OCR_captcha_recognition
```

#### Install dependencies
```bash
pip install -r requirements.txt
```

####  Training 
```bash
python src/train.py --epochs 50 --batch_size 32 --lr 0.001 --rnn_units 128
```
After training the model will be saved to models folders

#### Inference
```bash
python src/inference.py --images captcha_images_v2/2b827.png
```

#### Evaluate
```bash
python src/evaluate.py 
```
The metrics will be include:
- Sequence Accuracy (SeqAcc)
- Character Accuracy


### 🌐 API (FastAPI)
Start API server:
```bash
uvicorn app.app:app --reload --host 0.0.0.0 --port 8000
```

### 📌 UI (Streamlit)
```bash
streamlit run app/app_ui.py
```

### 🧪 Unit Test & API Test
Run tests:
```bash
pytest -v
```

### 🚀 Deployment (Docker)
Example Dockerfile:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY . .

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build & run:
```bash
docker build -t captcha-ocr .
docker run -p 8000:8000 captcha-ocr
```

---


## 👨‍💻 Author
OCR Captcha Project by **Ho Anh Khoi**

## 📜 License 
This project is licensed under the MIT License
