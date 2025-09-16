
import os, io
import streamlit as st
import pandas as pd
import numpy as np
import folium, cloudpickle as cp
from streamlit_folium import st_folium
from html import escape  # 변수명 안전 출력용

st.set_page_config(
    page_title="호수효과에 의한 서해안형 대설 1시간 적설량 예측 RandomForest 모델",
    layout="wide"
)

# --- 상단 우측 크레딧 ---
st.markdown(
    """
    <div style='text-align: right; color: gray; font-size: 0.9em;'>
        Developed by 전주기상지청 박성웅
    </div>
    """,
    unsafe_allow_html=True
)
# ------------------------

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
GLOBAL_MAE = float(METRICS.get("mae", pack.get("mae", 0.3)))

SRC = pack.get("source_meta", {})
# 파일 경로(있으면 사용)
PREVIEW_PATH = pack.get("preview_csv", None)
SOURCE_XLSX = pack.get("source_excel_path", None)
# pkl 내장 데이터(fallback)
PREVIEW_TEXT = pack.get("preview_csv_text", None)      # str
SOURCE_BYTES = pack.get("source_excel_bytes", None)    # bytes

# 웹페이지 표시에 사용할 변수 리스트(표시용 텍스트)
DISPLAY_BASIC = pack.get("display_basic_17", [])
DISPLAY_RULES = pack.get("display_rule_8", [])

# (선택) 버전 캡션
try:
    import sklearn
    st.caption(f"Runtime → scikit-learn {sklearn.__version__} / numpy {np.__version__}")
except Exception:
    pass

# ===== 제목 & 성능지표 =====
st.markdown("## 호수효과에 의한 서해안형 대설 1시간 적설량 예측 RandomForest 모델")
c1, c2, c3 = st.columns(3)
c1.metric("RMSE (cm/h)", f"{METRICS.get('rmse', float('nan')):.3f}" if METRICS else "—")
c2.metric("MAE (cm/h)", f"{METRICS.get('mae', float('nan')):.3f}" if METRICS else "—")
c3.metric("R²", f"{METRICS.get('r2', float('nan')):.3f}" if METRICS else "—")

# ===== 5개 지점(하드코딩) =====
SITES = {
    "군산": {"lat": 36.005, "lon": 126.761, "elev": 27.85},
    "정읍": {"lat": 35.563, "lon": 126.839, "elev": 68.70},
    "김제": {"lat": 35.809, "lon": 126.878, "elev": 54.61},
    "부안": {"lat": 35.730, "lon": 126.717, "elev": 12.20},
    "고창": {"lat": 35.348, "lon": 126.599, "elev": 52.42},
}
TARGET_SITES = list(SITES.keys())

# ===== 보조 함수: RH/Tw 계산 =====
def rh_from_T_Td(T, Td):
    es = 6.112 * np.exp((17.62 * T) / (243.12 + T))
    e  = 6.112 * np.exp((17.62 * Td) / (243.12 + Td))
    return float(np.clip(100.0 * (e / es), 0.0, 100.0))

def wetbulb_stull(T, RH):
    return float(
        T*np.arctan(0.151977*np.sqrt(RH+8.313659))
        + np.arctan(T+RH)
        - np.arctan(RH-1.676331)
        + 0.00391838*(RH**1.5)*np.arctan(0.023101*RH)
        - 4.686035
    )

# ===== 좌측 입력 · 우측 지도 =====
left, right = st.columns([1,2])

with left:
    st.subheader("🧾 입력값")

    # ▶ 지점/좌표 + 1시간 강수량
    mode = st.radio("입력 모드", ["5개 지점 자동", "사용자 좌표/지형 직접 입력"], horizontal=True)
    st.markdown("#### 📍 지점 · ☔ 1시간 강수량")
    c1, c2 = st.columns([1,1])

    with c1:
        if mode == "사용자 좌표/지형 직접 입력":
            site_selected = None
            lat = st.number_input("위도", 30.0, 45.0, 35.8, step=0.001)
            lon = st.number_input("경도", 120.0, 135.0, 126.8, step=0.001)
            elev = st.number_input("고도(m)", 0, 1500, 50, step=1)
        else:
            site_selected = st.selectbox("지점", TARGET_SITES, index=0)
            lat = SITES[site_selected]["lat"]
            lon = SITES[site_selected]["lon"]
            elev = SITES[site_selected]["elev"]

    with c2:
        prcp = st.number_input("1시간 강수량(mm)", 0.0, 50.0, 1.0, step=0.1)

    # ▶ 지상
    st.markdown("#### 🌡️ 지상")
    T      = st.number_input("기온(°C)", -30.0, 10.0, 0.0, step=0.1)
    Td     = st.number_input("이슬점온도(°C)", -30.0, 10.0, -5.0, step=0.1)
    wsp10  = st.number_input("10m 풍속(m/s)", 0.0, 40.0, 5.0, step=0.1)
    wdir10 = st.slider("10m 풍향(도)", 0, 360, 270, step=1)

    # ▶ 상층
    st.markdown("#### 🎈 상층")
    T850    = st.number_input("850hPa 기온(°C)", -40.0, 5.0, -7.0, step=0.1)
    T700    = st.number_input("700hPa 기온(°C)", -40.0, 0.0, -12.0, step=0.1)
    wdir850 = st.slider("850hPa 풍향(도)", 0, 360, 300, step=1)
    wsp850  = st.number_input("850hPa 풍속(m/s)", 0.0, 60.0, 15.0, step=0.1)

    # ▶ SST
    st.markdown("#### 🌊 해수면온도")
    sst = st.number_input("해수면온도(°C)", -2.0, 30.0, 6.0, step=0.1)

    # ▶ 파생 계산
    w10_sin, w10_cos   = np.sin(np.radians(wdir10)), np.cos(np.radians(wdir10))
    w850_sin, w850_cos = np.sin(np.radians(wdir850)), np.cos(np.radians(wdir850))
    RH = rh_from_T_Td(T, Td)
    Tw = wetbulb_stull(T, RH)
    KTS20_MS = 20 * 0.514444

    h700 = sst - T700
    if h700 <= 22: bin700 = 0
    elif h700 <= 25: bin700 = 1
    elif h700 <= 30: bin700 = 2
    else: bin700 = 3

    # ▶ 특징 벡터 구성 (X_cols에 맞춤)
    feat = {
        '기온(°C)': T, '이슬점온도(°C)': Td, '습구온도(°C)': Tw,
        '10m 풍속(m/s)': wsp10, '풍향_sin': w10_sin, '풍향_cos': w10_cos,
        '해수면온도(외연도)(°C)': sst, '해수면온도(부안)(°C)': sst,
        '850기온_평균(°C)': T850, '700기온_평균(°C)': T700,
        '850풍속_평균(m/s)': wsp850, '850풍향_sin': w850_sin, '850풍향_cos': w850_cos,
        '850hPa 해기차(°C)': sst - T850, '700hPa 해기차(°C)': h700,
        '규칙_풍향_300_330': int(300 <= wdir850 <= 330),
        '규칙_풍속_20kts이상': int(wsp850 >= KTS20_MS),
        '규칙_850T_영하8이하': int(T850 <= -8),
        '규칙_850해기차_20이상': int((sst - T850) >= 20),
        '1시간 강수량(mm)': prcp,
        '고도(m)': float(elev) if not pd.isna(elev) else 0.0,
    }
    feat['핵심규칙_충족'] = int(
        feat['규칙_풍향_300_330']==1 and
        feat['규칙_풍속_20kts이상']==1 and
        feat['규칙_850T_영하8이하']==1 and
        feat['규칙_850해기차_20이상']==1
    )
    feat['700해기차_bin']  = bin700
    feat['700해기차_점수'] = float(bin700)
    feat['규칙x700점수']  = float(feat['핵심규칙_충족']) * float(bin700)

    # ▶ 입력 행 생성 (결측 0으로 채움)
    row = {c: (feat[c] if c in feat else np.nan) for c in X_cols}
    X_input = pd.DataFrame([row], columns=X_cols).fillna(0.0)

    if st.button("예측 실행"):
        yhat = float(model.predict(X_input)[0])
        st.success(f"예측 적설량: **{yhat:.2f} cm/h** (±{GLOBAL_MAE:.2f})")

with right:
    st.subheader("🗺️ 예측 지도")

    def color_for(v):
        return "#1f77b4" if v < 1 else ("#2ca02c" if v < 3 else ("#ff7f0e" if v < 5 else "#d62728"))

    legend_html = """
    <div style="padding:8px 10px; background:#ffffff; border:1px solid #ddd; border-radius:8px; display:inline-block;">
      <div style="font-weight:600; margin-bottom:6px;">색상 기준 (cm/h)</div>
      <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block; width:14px; height:14px; background:#1f77b4; border:1px solid #333;"></span>
          <span>&lt; 1</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block; width:14px; height:14px; background:#2ca02c; border:1px solid #333;"></span>
          <span>1–3</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block; width:14px; height:14px; background:#ff7f0e; border:1px solid #333;"></span>
          <span>3–5</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block; width:14px; height:14px; background:#d62728; border:1px solid #333;"></span>
          <span>≥ 5</span>
        </div>
      </div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

    # 지도 생성
    m = folium.Map(location=[35.8, 126.9], zoom_start=8)

    # 5개 지점 일괄 예측
    results = []
    for name, info in SITES.items():
        row2 = {**row, '고도(m)': info['elev']}
        X2 = pd.DataFrame([row2], columns=X_cols).fillna(0.0)
        pred = float(model.predict(X2)[0])
        results.append((name, info["lat"], info["lon"], pred))

    for name, lat0, lon0, pred in results:
        col = color_for(pred)
        folium.CircleMarker(
            location=[lat0, lon0],
            radius=10,
            color=col, fill=True, fill_color=col, fill_opacity=0.9,
            tooltip=f"{name}: {pred:.2f} cm/h ±{GLOBAL_MAE:.2f}"
        ).add_to(m)

    st_folium(m, width=900, height=600)

# ===== 지도 하단: 모델 설명 =====
st.markdown("---")
st.markdown("#### 모델 설명")
st.write(
    "- **모델**: RandomForestRegressor (가이던스 파생변수 포함)\n"
    "- **타깃**: 1시간 신적설(cm)\n"
    "- **주요 입력**: 지상 T/Td/풍향·풍속, SST, 850/700hPa 기온/풍향·풍속, 해기차, 1시간 강수량, 고도 등\n"
    "- **평가 지표**: 상단의 RMSE/MAE/R²를 참고하세요.\n"
    "- **주의**: 입력 변수의 단위·컬럼명이 학습 시점과 달라지면 예측 정확도가 저하될 수 있습니다."
)

# ===== 사용된 엑셀 설명 + 데이터 구성(파란 박스) =====
st.markdown("#### 사용된 엑셀 설명")
if SRC:
    st.info(SRC.get("description", "업로드된 학습용 엑셀 데이터셋입니다."))
    st.write(f"- 파일명: **{SRC.get('source_filename','?')}**")
    st.write(f"- 크기: **{SRC.get('n_rows','?')} 행 × {SRC.get('n_cols','?')} 열**")
    # ✅ 데이터 구성(파란 박스)
    st.markdown("""
    <div style="background-color:#e7f1fb; padding:12px; border-radius:8px; border:1px solid #bcd0ef; line-height:1.6;">
      <b>📘 데이터 구성</b><br>
      - 서해안형 대설 발생 대표지점 5곳 기준<br>
      - 최근 5년간 사례 기반<br>
      - 서해안형 대설 발생 300건<br>
      - 서해안형 조건은 만족하지만 적설이 없는 사례 300건<br>
      - 서해안형 대설 기압계와 관계없는 사례 400건<br>
      <b>➡️ 총 1,000건의 데이터 학습 모델</b>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("엑셀 메타정보가 pack에 포함되지 않았습니다.")

# ===== 변수 목록: 온점(·) 구분 + 행간 개선 =====
# 공통 스타일 (온점, 줄간격, 줄바꿈 시 가독성)
st.markdown("""
<style>
.var-block{line-height:1.9; font-size:0.95rem;}
.var-line{margin: 6px 0 14px 0;}
.dot{opacity:0.55; padding: 0 8px;}
</style>
""", unsafe_allow_html=True)

def render_var_line(title, items):
    if not items:
        st.write("—")
        return
    # 항목 텍스트를 HTML-escape하고 온점으로 연결
    items_escaped = [escape(str(v)) for v in items]
    joined = f' <span class="dot">·</span> '.join(items_escaped)
    st.markdown(
        f'<div class="var-block"><b>{escape(title)}</b>'
        f'<div class="var-line">{joined}</div></div>',
        unsafe_allow_html=True
    )

render_var_line("🌍 기본 변수 (17개)", DISPLAY_BASIC)
render_var_line("📌 선행연구 규칙 기반 변수 (8개)", DISPLAY_RULES)

# ===== 엑셀 미리보기 (파일→내장텍스트 fallback) =====
st.markdown("**엑셀 미리보기(상위 50행)**")
if PREVIEW_PATH and os.path.exists(PREVIEW_PATH):
    prev = pd.read_csv(PREVIEW_PATH)
    st.dataframe(prev, use_container_width=True)
elif PREVIEW_TEXT:
    prev = pd.read_csv(io.StringIO(PREVIEW_TEXT))
    st.dataframe(prev, use_container_width=True)
else:
    st.info("미리보기 데이터가 없습니다.")

# ===== 데이터 다운로드 (파일→내장바이트/텍스트 fallback) =====
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
