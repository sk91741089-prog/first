
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
with open("snow_model.pkl","rb") as f:
    pack = cp.load(f)

model = pack["model"]
X_cols = pack["X_cols"]
METRICS = pack.get("metrics", {})
GLOBAL_MAE = float(METRICS.get("mae", 0.3))
SRC = pack.get("source_meta", {})
PREVIEW_TEXT = pack.get("preview_csv_text", None)
SOURCE_BYTES = pack.get("source_excel_bytes", None)
DATA_DESC = pack.get("data_desc", "")

# ===== 성능 지표 =====
st.markdown("## 호수효과에 의한 서해안형 대설 1시간 적설량 예측 RandomForest 모델")
c1, c2, c3 = st.columns(3)
c1.metric("RMSE (cm/h)", f"{METRICS.get('rmse', float('nan')):.3f}")
c2.metric("MAE (cm/h)", f"{METRICS.get('mae', float('nan')):.3f}")
c3.metric("R²", f"{METRICS.get('r2', float('nan')):.3f}")

# ===== 사용된 엑셀 설명 =====
st.markdown("#### 사용된 엑셀 설명")
if SRC:
    st.info(f"""
- 파일명: {SRC.get('source_filename','?')}
- 크기: {SRC.get('n_rows','?')} 행 × {SRC.get('n_cols','?')} 열
- 설명: {SRC.get('description','')}
---
{DATA_DESC}
""")
else:
    st.info("엑셀 메타정보 없음")

# ===== 미리보기 =====
st.markdown("**엑셀 미리보기(상위 50행)**")
if PREVIEW_TEXT:
    prev = pd.read_csv(io.StringIO(PREVIEW_TEXT))
    st.dataframe(prev, use_container_width=True)

# ===== 데이터 다운로드 =====
st.markdown("#### 데이터 다운로드")
if SOURCE_BYTES:
    st.download_button(
        "⬇️ 원본 엑셀 내려받기",
        data=SOURCE_BYTES,
        file_name=SRC.get("source_filename","uploaded_source.xlsx"),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
