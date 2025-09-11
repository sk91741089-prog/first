
import os, io
import streamlit as st
import pandas as pd
import numpy as np
import folium, cloudpickle as cp
from streamlit_folium import st_folium

st.set_page_config(
    page_title="호수효과에 의한 서해안형 대설 1시간 적설량 예측 RandomForest 모델",
    layout="wide"
)

# ===== 모델 로드 =====
try:
    with open("snow_model.pkl","rb") as f:
        pack = cp.load(f)
except Exception as e:
    st.error(f"모델/패키지 로드 오류: {e}\nrequirements.txt의 numpy/scikit-learn 버전을 피클 생성 환경과 맞춰주세요.")
    st.stop()

model = pack["model"]
X_cols = pack["X_cols"]

METRICS = pack.get("metrics", {})
GLOBAL_MAE = float(METRICS.get("mae", 0.3))

SRC = pack.get("source_meta", {})
# 파일 경로
PREVIEW_PATH = pack.get("preview_csv", None)
SOURCE_XLSX = pack.get("source_excel_path", None)
# 내장 데이터
PREVIEW_TEXT = pack.get("preview_csv_text", None)
SOURCE_BYTES = pack.get("source_excel_bytes", None)

# ===== 제목 & 성능지표 =====
st.markdown("## 호수효과에 의한 서해안형 대설 1시간 적설량 예측 RandomForest 모델")
c1, c2, c3 = st.columns(3)
c1.metric("RMSE (cm/h)", f"{METRICS.get('rmse', float('nan')):.3f}" if METRICS else "—")
c2.metric("MAE (cm/h)", f"{METRICS.get('mae', float('nan')):.3f}" if METRICS else "—")
c3.metric("R²", f"{METRICS.get('r2', float('nan')):.3f}" if METRICS else "—")

# ===== 업로드 엑셀 설명 + 데이터 구성 =====
st.markdown("#### 사용된 엑셀 설명")
if SRC:
    st.info(
        f"- 파일명: {SRC.get('source_filename','?')}\n"
        f"- 크기: {SRC.get('n_rows','?')} 행 × {SRC.get('n_cols','?')} 열\n"
        f"- 설명: {SRC.get('description','')}\n\n"
        "---\n"
        "📘 **데이터 구성**\n"
        "- 서해안형 대설 발생 대표지점 5곳 기준\n"
        "- 최근 5년간 사례 기반\n"
        "- 서해안형 대설 발생 300건\n"
        "- 조건 만족 무적설 300건\n"
        "- 무강설(기압계 무관) 400건\n"
        "➡ 총 1,000건의 빅데이터 학습 모델"
    )
else:
    st.info("엑셀 메타정보가 pack에 포함되지 않았습니다.")

# ===== 엑셀 미리보기 =====
st.markdown("**엑셀 미리보기(상위 50행)**")
if PREVIEW_PATH and os.path.exists(PREVIEW_PATH):
    prev = pd.read_csv(PREVIEW_PATH)
    st.dataframe(prev, use_container_width=True)
elif PREVIEW_TEXT:
    prev = pd.read_csv(io.StringIO(PREVIEW_TEXT))
    st.dataframe(prev, use_container_width=True)
else:
    st.info("미리보기 데이터가 없습니다.")

# ===== 데이터 다운로드 =====
st.markdown("#### 데이터 다운로드")
cA, cB = st.columns(2)

with cA:  # 원본 엑셀
    if SOURCE_XLSX and os.path.exists(SOURCE_XLSX):
        with open(SOURCE_XLSX, "rb") as f:
            st.download_button(
                "⬇️ 원본 엑셀 내려받기",
                data=f.read(),
                file_name=SRC.get("source_filename", "uploaded_source.xlsx"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    elif SOURCE_BYTES:
        st.download_button(
            "⬇️ 원본 엑셀 내려받기",
            data=SOURCE_BYTES,
            file_name=SRC.get("source_filename", "uploaded_source.xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        st.info("원본 엑셀 데이터가 없습니다.")

with cB:  # 미리보기 CSV
    if PREVIEW_PATH and os.path.exists(PREVIEW_PATH):
        with open(PREVIEW_PATH, "rb") as f:
            st.download_button(
                "⬇️ 미리보기 CSV 내려받기(상위 50행)",
                data=f.read(),
                file_name="uploaded_preview.csv",
                mime="text/csv",
                use_container_width=True,
            )
    elif PREVIEW_TEXT:
        st.download_button(
            "⬇️ 미리보기 CSV 내려받기(상위 50행)",
            data=PREVIEW_TEXT.encode("utf-8"),
            file_name="uploaded_preview.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("미리보기 CSV가 없습니다.")
