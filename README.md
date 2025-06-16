# ✍️ 손글씨 숫자 인식 앱 (MNIST 기반 ONNX 모델)

사용자가 웹 상에서 직접 숫자를 그리면, ONNX 형식의 MNIST 모델을 통해 해당 숫자가 무엇인지 예측해주는 **인터랙티브 웹 애플리케이션**입니다.  
Python의 Streamlit을 기반으로 개발되었으며, ONNX Runtime을 통해 경량화된 모델 추론을 빠르게 수행합니다.

<br>

## 🔍 주요 기능

1. **숫자 그리기** - 사용자가 자유롭게 숫자를 그림
2. **전처리 이미지 확인** - 모델 입력용 28x28 전처리 이미지 표시
3. **예측 결과 출력** - 예측 숫자 및 확률 표시 + 차트 시각화
4. **예측 기록 보기** - 지금까지 저장한 예측 이미지 및 결과 목록
![image](https://github.com/user-attachments/assets/31615194-9952-431e-beb4-fc184b906b9c)


<br>

## 📦 설치 및 실행 방법

### 1. 설치

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 실행

```bash
python -m streamlit run app.py
```

<br>


## 📁 프로젝트 구조

```
2025-mnist-canvas/
├── app.py                # Streamlit 애플리케이션
├── Dockerfile            # Docker 이미지 빌드 파일
├── mnist_model.onnx      # ONNX 모델
├── requirements.txt      # 필요한 패키지 목록
└── README.md             # 리드미 파일
```

<br>


## 🔗 참고

- [ONNX MNIST 모델](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist)
- [Streamlit 공식 문서](https://docs.streamlit.io/)
- [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas)

---
