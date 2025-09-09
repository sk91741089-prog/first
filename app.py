
import streamlit as st
import pandas as pd
import numpy as np
import folium, cloudpickle as cp
from streamlit_folium import st_folium

st.set_page_config(page_title="서해안 대설 1시간 적설량 예측", layout="wide")

# ===== 모델 로드 =====
with open("snow_model.pkl","rb") as f:
    pack = cp.load(f)
model = pack["model"]
X_cols = pack["X_cols"]
GLOBAL_MAE = float(pack.get("mae", 0.3))

# (선택) 런타임 버전 표시
try:
    import sklearn
    st.caption(f"Runtime → scikit-learn {sklearn.__version__} / numpy {np.__version__}")
except Exception:
    pass

st.markdown("### 호수효과 서해안 대설 **1시간 적설량 예측** (보조지표 웹앱)")
st.caption("모델: RandomForestRegressor (개선 전처리+가이던스 변수 포함) | 예측 표시: 값 ± MAE")

# ===== 5개 지점 (엑셀에서 산출해 하드코딩) =====
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
    """기온/이슬점으로 상대습도(%) 근사"""
    es = 6.112 * np.exp((17.62 * T) / (243.12 + T))
    e  = 6.112 * np.exp((17.62 * Td) / (243.12 + Td))
    return float(np.clip(100.0 * (e / es), 0.0, 100.0))

def wetbulb_stull(T, RH):
    """Stull(2011) 근사식으로 습구온도(°C) 계산"""
    return float(
        T*np.arctan(0.151977*np.sqrt(RH+8.313659))
        + np.arctan(T+RH)
        - np.arctan(RH-1.676331)
        + 0.00391838*(RH**1.5)*np.arctan(0.023101*RH)
        - 4.686035
    )

# ===== 좌측 입력, 우측 지도 =====
left, right = st.columns([1,2])

with left:
    st.subheader("🧾 입력값")

    # ▶ 최상단 묶음: 지점 + 1시간 강수량
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

    # (2) 지상
    st.markdown("#### 🌡️ 지상")
    T      = st.number_input("기온(°C)", -30.0, 10.0, 0.0, step=0.1)
    Td     = st.number_input("이슬점온도(°C)", -30.0, 10.0, -5.0, step=0.1)
    wsp10  = st.number_input("10m 풍속(m/s)", 0.0, 40.0, 5.0, step=0.1)
    wdir10 = st.slider("10m 풍향(도)", 0, 360, 270, step=1)

    # (3) 상층
    st.markdown("#### 🎈 상층")
    T850    = st.number_input("850hPa 기온(°C)", -40.0, 5.0, -7.0, step=0.1)
    T700    = st.number_input("700hPa 기온(°C)", -40.0, 0.0, -12.0, step=0.1)
    wdir850 = st.slider("850hPa 풍향(도)", 0, 360, 300, step=1)
    wsp850  = st.number_input("850hPa 풍속(m/s)", 0.0, 60.0, 15.0, step=0.1)

    # (4) 해수면온도 (단일 입력)
    st.markdown("#### 🌊 해수면온도")
    sst = st.number_input("해수면온도(°C)", -2.0, 30.0, 6.0, step=0.1)

    # 파생 계산
    w10_sin, w10_cos   = np.sin(np.radians(wdir10)), np.cos(np.radians(wdir10))
    w850_sin, w850_cos = np.sin(np.radians(wdir850)), np.cos(np.radians(wdir850))
    RH = rh_from_T_Td(T, Td)
    Tw = wetbulb_stull(T, RH)
    KTS20_MS = 20 * 0.514444

    # 700hPa 해기차 점수화
    h700 = sst - T700
    if h700 <= 22: bin700 = 0
    elif h700 <= 25: bin700 = 1
    elif h700 <= 30: bin700 = 2
    else: bin700 = 3

    # 특징 벡터 구성: X_cols에 맞춰 채움 (SST 단일 입력을 두 컬럼에 동일 주입)
    feat = {
        '기온(°C)': T,
        '이슬점온도(°C)': Td,
        '습구온도(°C)': Tw,
        '10m 풍속(m/s)': wsp10,
        '풍향_sin': w10_sin,
        '풍향_cos': w10_cos,

        '해수면온도(외연도)(°C)': sst,
        '해수면온도(부안)(°C)': sst,

        '850기온_평균(°C)': T850,
        '700기온_평균(°C)': T700,
        '850풍속_평균(m/s)': wsp850,
        '850풍향_sin': w850_sin,
        '850풍향_cos': w850_cos,

        '850hPa 해기차(°C)': sst - T850,
        '700hPa 해기차(°C)': h700,

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
    feat['700해기차_bin']   = bin700
    feat['700해기차_점수']  = float(bin700)
    feat['규칙x700점수']   = float(feat['핵심규칙_충족']) * float(bin700)

    # X_cols 순서대로 행 만들기 (없는 컬럼은 NaN) + 안전망: NaN -> 0
    row = {c: (feat[c] if c in feat else np.nan) for c in X_cols}
    X_input = pd.DataFrame([row], columns=X_cols)
    if X_input.isna().any().any():
        X_input = X_input.fillna(0.0)

    if st.button("예측 실행"):
        yhat = float(model.predict(X_input)[0])
        st.success(f"예측 적설량: **{yhat:.2f} cm/h** (±{GLOBAL_MAE:.2f})")

with right:
    st.subheader("🗺️ 예측 지도")

    # 색상 기준(범례: 실제 색상칩)
    def color_for(v):
        return "#1f77b4" if v < 1 else ("#2ca02c" if v < 3 else ("#ff7f0e" if v < 5 else "#d62728"))
    legend_html = """
    <div style="padding:8px 10px; background: #ffffff; border:1px solid #ddd; border-radius:8px; display:inline-block;">
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

    # 지도
    m = folium.Map(location=[35.8, 126.9], zoom_start=8)

    # 입력된 row를 기반으로 5개 지점 일괄 예측
    results = []
    for name, info in SITES.items():
        row2 = {**row}
        row2['고도(m)'] = info['elev']
        X2 = pd.DataFrame([row2], columns=X_cols)
        if X2.isna().any().any():
            X2 = X2.fillna(0.0)
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
