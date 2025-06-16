import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os
import requests
from streamlit_drawable_canvas import st_canvas

# 페이지 설정
st.set_page_config(
    page_title="숫자 인식 앱",
    page_icon="🔢",
    layout="wide"
)

# 세션 상태 초기화
if "history" not in st.session_state:
    st.session_state.history = []

# ONNX 모델 다운로드 함수
@st.cache_resource
def download_model():
    model_url = "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mnist/model/mnist-8.onnx"
    model_path = "mnist_model.onnx"
    
    if not os.path.exists(model_path):
        with st.spinner("MNIST 모델을 다운로드하는 중..."):
            response = requests.get(model_url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
            st.success("모델 다운로드 완료!")
    
    return model_path

# ONNX 모델 로드
@st.cache_resource
def load_model():
    model_path = download_model()
    session = ort.InferenceSession(model_path)
    return session

# 이미지 전처리 함수
def preprocess_image(image_data):
    if image_data is None:
        return None
    
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA')
    
    # 검은 배경으로 변환
    background = Image.new('RGB', img.size, (0, 0, 0))
    background.paste(img, mask=img.split()[0])
    img_gray = background.convert('L')
    
    # 리사이즈, 정규화
    img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)
    img_normalized = img_array.astype(np.float32) / 255.0
    img_final = img_normalized.reshape(1, 1, 28, 28)
    return img_final

# 예측 함수
def predict_digit(model, image):
    if image is None:
        return None, None
    
    input_name = model.get_inputs()[0].name
    result = model.run(None, {input_name: image})
    confidence = result[0][0]
    predicted_digit = np.argmax(confidence)

    def softmax(x):
        e_x = np.exp(x - np.max(x))  
        return e_x / e_x.sum()
    
    probabilities = softmax(confidence)
    return predicted_digit, probabilities

# 앱 제목
st.title("✍️ 손글씨 숫자 인식 앱")

# 모델 로드
try:
    model = load_model()
    st.success("MNIST 모델이 성공적으로 로드되었습니다!")
except Exception as e:
    st.error(f"모델 로드 실패: {str(e)}")
    st.stop()

# 3개 컬럼 구성
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.header("① 숫자를 그려보세요")
    canvas_result = st_canvas(
        stroke_width=15,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=250,
        height=250,
        drawing_mode="freedraw",
        key="canvas"
    )

with col2:
    st.header("② 전처리된 이미지")
    processed_image = None
    if canvas_result.image_data is not None:
        processed_image = preprocess_image(canvas_result.image_data)
        if processed_image is not None:
            processed_img_display = processed_image.reshape(28, 28)
            st.image(processed_img_display, caption="모델 입력 이미지 (28x28)", width=250)
        else:
            st.info("이미지를 처리하는 중...")
    else:
        st.info("숫자를 그려주세요.")

with col3:
    st.header("③ 예측 결과")
    if canvas_result.image_data is not None and processed_image is not None:
        predicted_digit, probabilities = predict_digit(model, processed_image)
        if predicted_digit is not None:
            st.subheader(f"예측 번호: {predicted_digit}")
            st.write(f"예측 확률: {probabilities[predicted_digit]:.2%}")
            st.subheader("각 숫자에 대한 예측 확률")
            st.bar_chart(probabilities)
        else:
            st.error("예측에 실패했습니다.")
    else:
        st.info("숫자를 그려주세요.")

    # 예측 결과 저장 버튼
    if st.button("💾 예측 결과 저장하기"):
        if canvas_result.image_data is not None and processed_image is not None:
            st.session_state.history.append({
                "original": canvas_result.image_data,
                "predicted": int(predicted_digit),
                "probability": float(probabilities[predicted_digit])
            })
            st.success("예측 결과가 저장되었습니다.")
        else:
            st.warning("예측된 결과가 없습니다.")

# 저장된 결과 영역
st.markdown("---")
st.header("④ 이미지 예측 기록")

# 초기화 버튼
if st.button("🗑️ 기록 초기화"):
    st.session_state.history.clear()
    st.success("예측 기록이 초기화되었습니다.")

if st.session_state.history:
    for i, item in enumerate(reversed(st.session_state.history)):
        col_img, col_info = st.columns([1, 2])
        with col_img:
            img = Image.fromarray(item["original"].astype("uint8"), "RGBA")
            st.image(img, width=100, caption=f"#{len(st.session_state.history)-i}")

        with col_info:
            st.write(f"예측 숫자: **{item['predicted']}**")
            st.write(f"확률: **{item['probability']:.2%}**")
else:
    st.info("저장된 예측 결과가 없습니다.")
