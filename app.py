import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="영남대 원룸 AI 중개사",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------------------------------------
# 2. 데이터 및 모델 로드 (최적화됨)
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    # 경로 설정
    model_dir = "models"

    # 모델 로드 (사용자 입력값 예측용으로 필요)
    voting_model = joblib.load(f"{model_dir}/voting_model.pkl")
    kpca = joblib.load(f"{model_dir}/kpca.pkl")
    kmeans = joblib.load(f"{model_dir}/kmeans.pkl")
    scaler_infra = joblib.load(f"{model_dir}/scaler_infra.pkl")
    scaler_final = joblib.load(f"{model_dir}/scaler_final.pkl")
    
    # 데이터 로드 (이미 PC1, PC2, cluster, lat, lon 등이 다 들어있음)
    df_normal = pd.read_csv("data/zigbang_streamlit.csv")
    
    # 2. 이상치 매물 파일 (새로 만든 것)
    # 파일명이 zicbang_only_outliers.csv 라고 하셨으므로 그대로 씁니다.
    df_outlier = pd.read_csv("data/zicbang_only_outliers.csv")


    # 위도/경도 이름 통일 (두 파일 모두 적용)
    for d in [df_normal, df_outlier]:
        if '위도' in d.columns: d.rename(columns={'위도': 'lat'}, inplace=True)
        if '경도' in d.columns: d.rename(columns={'경도': 'lon'}, inplace=True)
    
    # 리턴값에 df_outlier 추가
    return voting_model, kpca, kmeans, scaler_infra, scaler_final, df_normal, df_outlier

# 리소스 로드 및 데이터 분리
try:
    voting_model, kpca, kmeans, scaler_infra, scaler_final, df, df_outlier_all = load_resources()
    
except Exception as e:
    st.error(f"데이터 준비 중 오류 발생: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. 사이드바: 사용자 입력
# ---------------------------------------------------------
st.sidebar.title("🛠️ 내 방 조건 설정")
st.sidebar.markdown("원하는 방의 조건을 입력해주세요.")

# (1) 기본 정보 (정상 매물 df 기준)
min_dep, max_dep = int(df['보증금'].min()), int(df['보증금'].max())
min_area, max_area = float(df['전용면적'].min()), float(df['전용면적'].max())
min_floor, max_floor = int(df['해당층'].min()), int(df['해당층'].max())
min_age, max_age = int(df['노후도'].min()), int(df['노후도'].max())

# 슬라이더 설정
deposit = st.sidebar.slider("보증금 (만원) [이하]", min_dep, max_dep, max_dep, step=10)
area = st.sidebar.slider("전용면적 (m²) [이상]", min_area, max_area, min_area)
floor = st.sidebar.slider("층수 [이상]", min_floor, max_floor, min_floor)
age = st.sidebar.slider("건물 노후도 (년) [이하]", min_age, max_age, max_age)

# (2) 옵션 정보
st.sidebar.subheader("옵션")
col1, col2, col3 = st.sidebar.columns(3)
elevator = st.sidebar.checkbox("엘리베이터", value=False)
south = st.sidebar.checkbox("남향", value=False)
full_opt = st.sidebar.checkbox("풀옵션", value=False)

# (3) 인프라 거리 정보
st.sidebar.subheader("주변 편의시설 거리 (m) [이하]")

max_subway = int(df['지하철역_거리(m)'].max())
max_bus = int(df['버스정류장_거리(m)'].max())
max_mart = int(df['대형마트_거리(m)'].max())
max_conv = int(df['편의점_거리(m)'].max())
max_laundry = int(df['세탁소_거리(m)'].max())
max_cafe = int(df['카페_거리(m)'].max())
max_pharm = int(df['약국_거리(m)'].max())

subway = st.sidebar.slider("지하철역", 0, max_subway, max_subway)
bus = st.sidebar.slider("버스정류장", 0, max_bus, max_bus)
mart = st.sidebar.slider("대형마트", 0, max_mart, max_mart)
conv = st.sidebar.slider("편의점", 0, max_conv, max_conv)
laundry = st.sidebar.slider("세탁소", 0, max_laundry, max_laundry)
cafe = st.sidebar.slider("카페", 0, max_cafe, max_cafe)
pharmacy = st.sidebar.slider("약국", 0, max_pharm, max_pharm)

# ---------------------------------------------------------
# 4. 예측 로직
# ---------------------------------------------------------
if st.sidebar.button("💰 월세 예측 & 매물 찾기", type="primary"):
    
    # Step 1: PCA (사용자 입력값 변환)
    infra_input = pd.DataFrame([[laundry, cafe, pharmacy, mart, conv]], 
                               columns=['세탁소_거리(m)', '카페_거리(m)', '약국_거리(m)', '대형마트_거리(m)', '편의점_거리(m)'])
    infra_scaled = scaler_infra.transform(infra_input)
    pca_res = kpca.transform(infra_scaled)
    pc1, pc2 = pca_res[0][0], pca_res[0][1]
    
    # Step 2: 군집화 (사용자 입력값 변환)
    cluster_features = pd.DataFrame([[
        deposit, area, age, floor, int(elevator), int(south), int(full_opt), pc1, pc2, subway, bus
    ]], columns=['보증금', '전용면적', '노후도', '해당층', '엘리베이터', '남향', '풀옵션', 'PC1', 'PC2', '지하철역_거리(m)', '버스정류장_거리(m)'])
    
    cluster_scaled = scaler_final.transform(cluster_features)
    predicted_cluster = kmeans.predict(cluster_scaled)[0]
    
    # Step 3: 월세 예측
    input_data = {
        '보증금': deposit, 
        '전용면적': area, 
        '노후도': age, 
        '해당층': floor, 
        '엘리베이터': elevator, 
        '남향': int(south), 
        '풀옵션': int(full_opt),
        'PC1': pc1, 
        'PC2': pc2, 
        'cluster_0': 1 if predicted_cluster == 0 else 0,
        'cluster_1': 1 if predicted_cluster == 1 else 0,
        'cluster_2': 1 if predicted_cluster == 2 else 0,
        'cluster_3': 1 if predicted_cluster == 3 else 0,
        '지하철역_거리(m)' : subway,
        '버스정류장_거리(m)' : bus
    }
    
    final_cols = ['보증금', '전용면적', '노후도', '해당층', '엘리베이터', '남향', '풀옵션', 'PC1', 'PC2', 
                  'cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', '지하철역_거리(m)', '버스정류장_거리(m)']
    
    model_input_df = pd.DataFrame([input_data])[final_cols]
    predicted_price = voting_model.predict(model_input_df)[0]
    
    # 결과 저장
    st.session_state['predicted_price'] = predicted_price
    st.session_state['predicted_cluster'] = predicted_cluster
    st.session_state['prediction_done'] = True

# ---------------------------------------------------------
# 5. 결과 화면 출력
# ---------------------------------------------------------
if st.session_state.get('prediction_done'):
    
    price = st.session_state['predicted_price']
    cluster = st.session_state['predicted_cluster']
    
    st.header(f"🏠 AI가 분석한 적정 월세: :blue[{price:.1f}만원]")
    st.write(f"당신의 조건은 **'군집 {cluster}번'** 유형에 속합니다.")
    st.divider()
    
    # 공통 필터 함수
    def apply_filter(target_df, ignore_deposit=False):
        mask = (
            (target_df['보증금'] <= deposit) &
            (target_df['전용면적'] >= area) &
            (target_df['해당층'] >= floor) &
            (target_df['노후도'] <= age) &
            (target_df['엘리베이터'] == int(elevator)) &
            (target_df['남향'] == int(south)) &
            (target_df['풀옵션'] == int(full_opt)) &
            (target_df['지하철역_거리(m)'] <= subway) &
            (target_df['버스정류장_거리(m)'] <= bus) &
            (target_df['대형마트_거리(m)'] <= mart) &
            (target_df['편의점_거리(m)'] <= conv) &
            (target_df['세탁소_거리(m)'] <= laundry) &
            (target_df['카페_거리(m)'] <= cafe) &
            (target_df['약국_거리(m)'] <= pharmacy)
        )

        if not ignore_deposit:
                    mask = mask & (target_df['보증금'] <= deposit)
                    
        return target_df[mask].copy()
            
    # 데이터 나누기 및 필터링
    # 이미 저장된 CSV에 int 변환 등이 되어 있으므로 바로 필터 적용 가능
    recommendations = apply_filter(df)
    
    # (B) 이상치 매물: ignore_deposit=True -> 보증금 8000만원짜리도 나옴!
    outliers = apply_filter(df_outlier_all, ignore_deposit=True)
    
    # 정렬
    recommendations = recommendations.sort_values(by='월세', ascending=True)
    outliers = outliers.sort_values(by='월세', ascending=True)

    # (A) 정상 추천 매물 출력
    st.subheader("🔍 조건 완전 충족 매물 (월세 낮은 순)")
    
    col_map, col_list = st.columns([2, 1])
    
    with col_map:
        if not recommendations.empty:
            st.write(f"조건을 만족하는 **{len(recommendations)}개**의 정상 매물을 찾았습니다.")
            
            recommendations['color'] = [[0, 0, 255, 140] for _ in range(len(recommendations))]
            recommendations['size'] = 50
            
            st.map(recommendations, color='color', size='size', zoom=14)
        else:
            st.warning("조건을 만족하는 정상 매물이 없습니다.")
            st.info("💡 팁: 거리를 조금 늘리거나 조건을 완화해보세요.")

    with col_list:
        if not recommendations.empty:
            st.write("📋 **정상 매물 리스트**")
            display_cols = ['월세', '보증금', '전용면적', '해당층']
            st.dataframe(
                recommendations[display_cols].style.format({
                    '월세': '{:.1f}만', '보증금': '{}만', '전용면적': '{:.1f}m²'
                }),
                height=500,
                hide_index=True
            )
        else:
            st.write("매물이 없습니다.")

    st.divider()

    # (B) 이상치(위험/허위) 매물 출력
    st.subheader("🚨 주의! 조건은 맞지만 '이상한 매물(Outlier)'")
    st.error("이 매물들은 고객님의 조건에는 맞지만, 시세보다 지나치게 싸거나 비싼 매물입니다. 계약 시 주의하세요!")
    
    if not outliers.empty:
        col_out1, col_out2 = st.columns([2, 1])     
        with col_out1:
             st.write(f"주의가 필요한 매물 **{len(outliers)}개**가 발견되었습니다.")
             
             outliers['color'] = [[255, 0, 0, 140] for _ in range(len(outliers))]
             outliers['size'] = 50
             
             st.map(outliers, color='color', size='size', zoom=14)
             
        with col_out2:
             st.write("📋 **주의 매물 리스트**")
             
             # [수정] 이상치 사유를 한글로 보기 좋게 변경
             # 예: "월세_high" -> "월세(높음)", "전용면적_low" -> "전용면적(낮음)"
             if 'outlier_reason' in outliers.columns:
                 outliers['이상치 사유'] = outliers['outlier_reason'].astype(str).str.replace('_high', '(높음)').str.replace('_low', '(낮음)')
             else:
                 outliers['이상치 사유'] = "정보 없음"
             
             # [수정] 표에 '이상치 사유' 컬럼 추가
             display_cols = ['월세', '보증금', '전용면적', '해당층', '이상치 사유']
             
             st.dataframe(
                outliers[display_cols].style.format({
                    '월세': '{:.1f}만', '보증금': '{}만', '전용면적': '{:.1f}m²'
                }),
                height=500,
                hide_index=True
            )
    else:
        st.success("다행히 조건에 맞는 매물 중 '위험 매물'은 발견되지 않았습니다. 안심하세요! 🛡️")
else:
    st.info("👈 왼쪽 사이드바에서 조건을 설정하고 '예측하기' 버튼을 눌러주세요.")