import streamlit as st
import requests

API_URL_SINGLE = "http://localhost:8000/predict"
API_URL_BATCH = "http://localhost:8000/predict_batch"

st.set_page_config(page_title="Captcha OCR", page_icon="🔑", layout="wide")

def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(120deg, #e0f7fa, #ede7f6);
            background-attachment: fixed;
        }
        .block-container {
            padding: 2rem;
            border-radius: 16px;
            background-color: rgba(255,255,255,0.95);
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        h1, h2, h3 {
            color: #4a148c;
        }
        .avatar-img {
            border-radius: 50%;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def single_prediction():
    st.subheader("📌 Single Prediction")
    st.write("Upload **one captcha image** and get the recognized text.")

    uploaded_file = st.file_uploader("Upload one captcha image", type=["png", "jpg", "jpeg"], key="single")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Captcha", use_container_width=True)

        if st.button("🔍 Predict Single"):
            with st.spinner("🔄 Processing..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(API_URL_SINGLE, files=files)

            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ Captcha Text: **{result['captcha_text']}**")
            else:
                st.error(f"❌ Error: {response.text}")

def batch_prediction():
    st.subheader("📌 Batch Prediction")
    st.write("Upload **multiple captcha images** and get all results in one shot.")

    uploaded_files = st.file_uploader(
        "Upload multiple captcha images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="batch"
    )

    if uploaded_files:
        st.write(f"🖼 Uploaded **{len(uploaded_files)}** images")

        if st.button("🔍 Predict Batch"):
            with st.spinner("🔄 Processing batch..."):
                files = [("files", (f.name, f, f.type)) for f in uploaded_files]
                response = requests.post(API_URL_BATCH, files=files)

            if response.status_code == 200:
                result = response.json()
                st.success("✅ Batch prediction results:")
                st.dataframe(result["results"])
            else:
                st.error(f"❌ Error: {response.text}")

def main():
    set_background()

    left, center, right = st.columns([1, 2, 1])

    with left:
        st.image("https://cdn-icons-png.flaticon.com/512/2920/2920244.png", width=100)
        st.markdown("### 🎨 Decoration")
        st.write("🔑 OCR Captcha")  
        st.write("🤖 Powered by CRNN+CTC")  
        st.write("✨ FastAPI + Streamlit")  

    with center:
        st.title("🔑 OCR Captcha API Web")
        st.markdown(
            """
            Welcome to the **OCR Captcha Recognition Web APP** 🎉  
            This web app uses **CRNN + CTC** model to decode captcha images.  
            Choose a mode below to start:
            """,
            unsafe_allow_html=True,
        )
        tab1, tab2 = st.tabs(["🖼 Single Prediction", "📂 Batch Prediction"])
        with tab1:
            single_prediction()
        with tab2:
            batch_prediction()

    with right:
        st.markdown("### 👨‍💻 Author")
        st.image("app/me.jpg", width=100)
        st.write("**Ho Anh Khoi**")
        st.write("📦 Docker-ready")  
        st.write("🧪 Tested with Pytest")  
        st.write("🌐 API: FastAPI")  

if __name__ == "__main__":
    main()
