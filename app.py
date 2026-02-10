import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------------------------------------
# 1. 페이지 설정 (가장 먼저 실행되어야 함)
# ---------------------------------------------------------
st.set_page_config(
    page_title="영남대 원룸 AI 중개사",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------------------------------------
# 2. 데이터 및 모델 로드 함수 (캐싱 적용으로 속도 향상)
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    # 경로 설정 (폴더 구조에 맞게)
    model_dir = "models"
    data_path = "data/zigbang_outlier.csv"
    
    # 모델 로드
    voting_model = joblib.load(f"{model_dir}/voting_model.pkl")
    kpca = joblib.load(f"{model_dir}/kpca.pkl")
    kmeans = joblib.load(f"{model_dir}/kmeans.pkl")
    scaler_infra = joblib.load(f"{model_dir}/scaler_infra.pkl")
    scaler_final = joblib.load(f"{model_dir}/scaler_final.pkl")
    
    # 데이터 로드
    df = pd.read_csv(data_path)
    
    # 지도 표시를 위해 위도/경도 컬럼명 변경 (Streamlit 표준: lat, lon)
    if '위도' in df.columns and '경도' in df.columns:
        df = df.rename(columns={'위도': 'lat', '경도': 'lon'})
        
    return voting_model, kpca, kmeans, scaler_infra, scaler_final, df

# 리소스 로드 실행
try:
    voting_model, kpca, kmeans, scaler_infra, scaler_final, df = load_resources()
    # ==============================================================================
    # [긴급 추가] 불러온 데이터(df)에는 'cluster' 정보가 없으므로, 여기서 계산해서 넣어줍니다.
    # ==============================================================================
    
    # 1. 인프라 PCA 변환 (전체 데이터에 대해 수행)
    infra_cols = ['세탁소_거리(m)', '카페_거리(m)', '약국_거리(m)', '대형마트_거리(m)', '편의점_거리(m)']
    infra_scaled_df = scaler_infra.transform(df[infra_cols])
    pca_res_df = kpca.transform(infra_scaled_df)
    
    df['PC1'] = pca_res_df[:, 0]
    df['PC2'] = pca_res_df[:, 1]
    
    # 2. 불리언 타입(True/False)을 숫자(1/0)로 변환
    for col in ['엘리베이터', '남향', '풀옵션']:
        if col in df.columns:
            df[col] = df[col].astype(int)
            
    # 3. 군집화 수행 (전체 데이터에 대해 수행)
    # 학습할 때 썼던 변수 순서 그대로 가져와야 합니다.
    features_for_clustering = [
        '보증금', '전용면적', '노후도', '해당층', 
        '엘리베이터', '남향', '풀옵션', 'PC1', 'PC2', 
        '지하철역_거리(m)', '버스정류장_거리(m)'
    ]
    
    cluster_scaled_df = scaler_final.transform(df[features_for_clustering])
    df['cluster'] = kmeans.predict(cluster_scaled_df)

except Exception as e:
    st.error(f"파일을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. 사이드바: 사용자 입력 받기
# ---------------------------------------------------------
st.sidebar.title("🛠️ 내 방 조건 설정")
st.sidebar.markdown("원하는 방의 조건을 입력해주세요.")

# (1) 기본 정보
st.sidebar.subheader("기본 정보")
deposit = st.sidebar.slider("보증금 (만원)", 100, 5000, 300, step=50)
area = st.sidebar.slider("전용면적 (m²)", 10.0, 66.0, 25.0)
floor = st.sidebar.slider("층수", 1, 20, 3)
age = st.sidebar.slider("건물 노후도 (년)", 0, 35, 5)

# (2) 옵션 정보
st.sidebar.subheader("옵션")
col1, col2, col3 = st.sidebar.columns(3)
elevator = st.sidebar.checkbox("엘리베이터", value=True)
south = st.sidebar.checkbox("남향", value=False)
full_opt = st.sidebar.checkbox("풀옵션", value=True)

# (3) 인프라 거리 정보 (PCA용)
st.sidebar.subheader("주변 편의시설 거리 (m)")
subway = st.sidebar.slider("지하철역", 0, 2000, 500)
bus = st.sidebar.slider("버스정류장", 0, 1000, 150)
mart = st.sidebar.slider("대형마트", 0, 3000, 1000)
conv = st.sidebar.slider("편의점", 0, 1000, 100)
laundry = st.sidebar.slider("세탁소", 0, 1000, 200)
cafe = st.sidebar.slider("카페", 0, 1000, 100)
pharmacy = st.sidebar.slider("약국", 0, 1000, 200)

# ---------------------------------------------------------
# 4. 예측 로직
# ---------------------------------------------------------
if st.sidebar.button("💰 월세 예측 & 매물 찾기", type="primary"):
    
    # --- [Step 1] 인프라 PCA 변환 ---
    # 학습 때 사용한 컬럼 순서 정확히 지키기
    infra_input = pd.DataFrame([[laundry, cafe, pharmacy, mart, conv]], 
                               columns=['세탁소_거리(m)', '카페_거리(m)', '약국_거리(m)', '대형마트_거리(m)', '편의점_거리(m)'])
    
    infra_scaled = scaler_infra.transform(infra_input)
    pca_res = kpca.transform(infra_scaled) # PC1, PC2 추출
    
    pc1, pc2 = pca_res[0][0], pca_res[0][1]
    
    # --- [Step 2] 군집(Cluster) 예측 ---
    # 군집화 학습 때 사용한 변수 순서대로 구성
    cluster_features = pd.DataFrame([[
        deposit, area, age, floor, elevator, int(south), int(full_opt), pc1, pc2, subway, bus
    ]], columns=['보증금', '전용면적', '노후도', '해당층', '엘리베이터', '남향', '풀옵션', 'PC1', 'PC2', '지하철역_거리(m)', '버스정류장_거리(m)'])
    
    # 군집화용 스케일링
    cluster_scaled = scaler_final.transform(cluster_features)
    predicted_cluster = kmeans.predict(cluster_scaled)[0]
    
    # --- [Step 3] 최종 월세 예측 (Voting Model) ---
    # 학습 때 사용한 최종 변수 순서 및 Cluster One-Hot Encoding
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
    
    # 데이터프레임 변환 (학습 시 컬럼 순서 보장을 위해 재정렬 필요할 수 있음)
    # 모델 학습 시 사용한 features 리스트 순서와 동일하게 정렬
    final_cols = ['보증금', '전용면적', '노후도', '해당층', '엘리베이터', '남향', '풀옵션', 'PC1', 'PC2', 
                  'cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', '지하철역_거리(m)', '버스정류장_거리(m)']
    
    # 만약 학습 모델 컬럼과 다르다면 에러가 날 수 있으니, features 순서 중요
    model_input_df = pd.DataFrame([input_data])[final_cols]
    
    predicted_price = voting_model.predict(model_input_df)[0]
    
    # ---------------------------------------------------------
    # 5. 결과 화면 구성
    # ---------------------------------------------------------
    st.header(f"🏠 AI가 분석한 적정 월세: :blue[{predicted_price:.1f}만원]")
    st.write(f"당신의 조건은 **'군집 {predicted_cluster}번'** 유형에 속합니다.")
    
    st.divider()
    
    # ---------------------------------------------------------
    # 6. 추천 시스템 & 7. 이상치 확인 (통합 필터링)
    # ---------------------------------------------------------
    
    # [1] 공통 필터 정의 (추천 & 이상치 모두 적용)
    # : 사용자가 설정한 모든 조건(가격, 스펙, 옵션, 거리)을 만족하는지 확인
    mask_common = (
        (df['보증금'] <= deposit) &
        (df['전용면적'] >= area) &
        (df['해당층'] >= floor) &
        (df['노후도'] <= age) &
        (df['엘리베이터'] == int(elevator)) &
        (df['남향'] == int(south)) &
        (df['풀옵션'] == int(full_opt)) &
        (df['지하철역_거리(m)'] <= subway) &
        (df['버스정류장_거리(m)'] <= bus) &
        (df['대형마트_거리(m)'] <= mart) &
        (df['편의점_거리(m)'] <= conv) &
        (df['세탁소_거리(m)'] <= laundry) &
        (df['카페_거리(m)'] <= cafe) &
        (df['약국_거리(m)'] <= pharmacy)
    )
    
    # [2] 데이터 나누기
    # - 추천 매물: 공통 조건 만족 + 정상 매물(is_outlier == False)
    # - 위험 매물: 공통 조건 만족 + 이상치 매물(is_outlier == True)
    recommendations = df[mask_common & (df['is_outlier'] == False)].copy()
    outliers = df[mask_common & (df['is_outlier'] == True)].copy()
    
    # 정렬 (월세 싼 순서)
    recommendations = recommendations.sort_values(by='월세', ascending=True)
    outliers = outliers.sort_values(by='월세', ascending=True)

    # =========================================================
    # (A) 정상 추천 매물 출력
    # =========================================================
    st.subheader("🔍 조건 완전 충족 매물 (월세 낮은 순)")
    
    col_map, col_list = st.columns([2, 1])
    
    with col_map:
        if not recommendations.empty:
            st.write(f"조건을 만족하는 **{len(recommendations)}개**의 정상 매물을 찾았습니다.")
            
            # 파란색 투명 마커
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

    # =========================================================
    # (B) 이상치(위험/허위) 매물 출력
    # =========================================================
    st.subheader("🚨 주의! 조건은 맞지만 '이상한 매물(Outlier)'")
    st.error("이 매물들은 고객님의 조건에는 맞지만, 시세보다 지나치게 싸거나(허위매물 의심) 비싼 매물입니다. 계약 시 주의하세요!")
    
    if not outliers.empty:
        col_out1, col_out2 = st.columns([2, 1])     
        with col_out1:
             st.write(f"주의가 필요한 매물 **{len(outliers)}개**가 발견되었습니다.")
             
             # 빨간색 투명 마커 (위험 신호!)
             outliers['color'] = [[255, 0, 0, 140] for _ in range(len(outliers))]
             outliers['size'] = 50
             
             st.map(outliers, color='color', size='size', zoom=14)
             
        with col_out2:
             st.write("📋 **주의 매물 리스트**")
             display_cols = ['월세', '보증금', '전용면적', '해당층']
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