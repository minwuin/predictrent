import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Yeungnam Univ. Studio AI Agent",
    page_icon="🏠",
    layout="wide"
)
if 'prediction_done' not in st.session_state:
    st.session_state['prediction_done'] = False
if 'predicted_price' not in st.session_state:
    st.session_state['predicted_price'] = 0.0
if 'predicted_cluster' not in st.session_state:
    st.session_state['predicted_cluster'] = 0
# ---------------------------------------------------------
# 2. Load Resources (Optimized)
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    # Path settings
    model_dir = "models"

    # Load models
    voting_model = joblib.load(f"{model_dir}/voting_model.pkl")
    kpca = joblib.load(f"{model_dir}/kpca.pkl")
    kmeans = joblib.load(f"{model_dir}/kmeans.pkl")
    scaler_infra = joblib.load(f"{model_dir}/scaler_infra.pkl")
    scaler_final = joblib.load(f"{model_dir}/scaler_final.pkl")
    
    # Load data (Normal listings)
    df_normal = pd.read_csv("data/zigbang_streamlit.csv")
    
    # Load outlier data
    df_outlier = pd.read_csv("data/zicbang_only_outliers.csv")

    # Unify Latitude/Longitude names
    for d in [df_normal, df_outlier]:
        if '위도' in d.columns: d.rename(columns={'위도': 'lat'}, inplace=True)
        if '경도' in d.columns: d.rename(columns={'경도': 'lon'}, inplace=True)
    
    return voting_model, kpca, kmeans, scaler_infra, scaler_final, df_normal, df_outlier

# Resource Loading
try:
    voting_model, kpca, kmeans, scaler_infra, scaler_final, df, df_outlier_all = load_resources()
    
except Exception as e:
    st.error(f"Error preparing data: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. Sidebar: User Input
# ---------------------------------------------------------
st.sidebar.title("🛠️ My Room Preferences")
st.sidebar.markdown("Please enter your desired room conditions.")

# (1) Basic Information
min_dep, max_dep = int(df['보증금'].min()), int(df['보증금'].max())
min_area, max_area = float(df['전용면적'].min()), float(df['전용면적'].max())
min_floor, max_floor = int(df['해당층'].min()), int(df['해당층'].max())
min_age, max_age = int(df['노후도'].min()), int(df['노후도'].max())

# Sliders
deposit = st.sidebar.slider("Deposit (10k KRW) [Max]", min_dep, max_dep, max_dep, step=10)
area = st.sidebar.slider("Net Area (m²) [Min]", min_area, max_area, min_area)
floor = st.sidebar.slider("Floor [Min]", min_floor, max_floor, min_floor)
age = st.sidebar.slider("Building Age (Years) [Max]", min_age, max_age, max_age)

# (2) Options
st.sidebar.subheader("Options")
col1, col2, col3 = st.sidebar.columns(3)
elevator = st.sidebar.checkbox("Elevator", value=False)
south = st.sidebar.checkbox("South-Facing", value=False)
full_opt = st.sidebar.checkbox("Full Option", value=False)

# (3) Infrastructure Distance Info
st.sidebar.subheader("Nearby Amenities Distance (m) [Max]")

max_subway = int(df['지하철역_거리(m)'].max())
max_bus = int(df['버스정류장_거리(m)'].max())
max_mart = int(df['대형마트_거리(m)'].max())
max_conv = int(df['편의점_거리(m)'].max())
max_laundry = int(df['세탁소_거리(m)'].max())
max_cafe = int(df['카페_거리(m)'].max())
max_pharm = int(df['약국_거리(m)'].max())

subway = st.sidebar.slider("Subway Station", 0, max_subway, max_subway)
bus = st.sidebar.slider("Bus Stop", 0, max_bus, max_bus)
mart = st.sidebar.slider("Supermarket", 0, max_mart, max_mart)
conv = st.sidebar.slider("Convenience Store", 0, max_conv, max_conv)
laundry = st.sidebar.slider("Laundry", 0, max_laundry, max_laundry)
cafe = st.sidebar.slider("Cafe", 0, max_cafe, max_cafe)
pharmacy = st.sidebar.slider("Pharmacy", 0, max_pharm, max_pharm)

# ---------------------------------------------------------
# 4. Prediction Logic
# ---------------------------------------------------------
if st.sidebar.button("💰 Predict Rent & Find Rooms", type="primary"):
    
    # Step 1: PCA
    infra_input = pd.DataFrame([[laundry, cafe, pharmacy, mart, conv]], 
                               columns=['세탁소_거리(m)', '카페_거리(m)', '약국_거리(m)', '대형마트_거리(m)', '편의점_거리(m)'])
    infra_scaled = scaler_infra.transform(infra_input)
    pca_res = kpca.transform(infra_scaled)
    pc1, pc2 = pca_res[0][0], pca_res[0][1]
    
    # Step 2: Clustering
    cluster_features = pd.DataFrame([[
        deposit, area, age, floor, int(elevator), int(south), int(full_opt), pc1, pc2, subway, bus
    ]], columns=['보증금', '전용면적', '노후도', '해당층', '엘리베이터', '남향', '풀옵션', 'PC1', 'PC2', '지하철역_거리(m)', '버스정류장_거리(m)'])
    
    cluster_scaled = scaler_final.transform(cluster_features)
    predicted_cluster = kmeans.predict(cluster_scaled)[0]
    
    # Step 3: Rent Prediction
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
    
    # Store results
    st.session_state['predicted_price'] = predicted_price
    st.session_state['predicted_cluster'] = predicted_cluster
    st.session_state['prediction_done'] = True

# ---------------------------------------------------------
# 5. Output Results
# ---------------------------------------------------------
if st.session_state.get('prediction_done'):
    
    price = st.session_state['predicted_price']
    cluster = st.session_state['predicted_cluster']
    
    st.header(f"🏠 AI Estimated Fair Rent: :blue[{price:.1f}k KRW]")
    st.write(f"Your criteria belong to the **'Cluster {cluster}'** type.")
    st.divider()
    
    # Common Filter Function
    def apply_filter(target_df, ignore_deposit=False):
        mask = (
            (target_df['보증금'] <= deposit) &
            (target_df['전용면적'] >= area) &
            (target_df['해당층'] >= floor) &
            (target_df['노후도'] <= age) &
            (target_df['지하철역_거리(m)'] <= subway) &
            (target_df['버스정류장_거리(m)'] <= bus) &
            (target_df['대형마트_거리(m)'] <= mart) &
            (target_df['편의점_거리(m)'] <= conv) &
            (target_df['세탁소_거리(m)'] <= laundry) &
            (target_df['카페_거리(m)'] <= cafe) &
            (target_df['약국_거리(m)'] <= pharmacy)
        )
        
        if elevator:
            mask = mask & (target_df['엘리베이터'] == 1)
        if south:
            mask = mask & (target_df['남향'] == 1)
        if full_opt:
            mask = mask & (target_df['풀옵션'] == 1)

        if not ignore_deposit:
            mask = mask & (target_df['보증금'] <= deposit)
                    
        return target_df[mask].copy()
            
    # Data filtering
    recommendations = apply_filter(df)
    if not recommendations.empty:
        recommendations = recommendations[recommendations['월세'] <= price]
    
    # (B) Outlier listings
    outliers = apply_filter(df_outlier_all, ignore_deposit=True)
    
    # Sorting
    recommendations = recommendations.sort_values(by='월세', ascending=True)
    outliers = outliers.sort_values(by='월세', ascending=True)

    # (A) Normal Recommended Listings
    st.subheader("🔍 Matching Listings (Sorted by Lowest Rent)")
    
    col_map, col_list = st.columns([2, 1])
    
    with col_map:
        if not recommendations.empty:
            st.write(f"Found **{len(recommendations)}** normal listings that match your criteria.")
            
            recommendations['color'] = [[0, 0, 255, 140] for _ in range(len(recommendations))]
            recommendations['size'] = 50
            
            st.map(recommendations, color='color', size='size', zoom=14)
        else:
            st.warning("No normal listings found matching your criteria.")
            st.info("💡 Tip: Try increasing the distance or loosening your requirements.")

    with col_list:
        if not recommendations.empty:
            st.write("📋 **Normal Listing List**")
            # Using original Korean column names for internal logic, mapping for display
            display_cols = ['월세', '보증금', '전용면적', '해당층']
            st.dataframe(
                recommendations[display_cols].rename(columns={
                    '월세': 'Rent', '보증금': 'Deposit', '전용면적': 'Area', '해당층': 'Floor'
                }).style.format({
                    'Rent': '{:.1f}k', 'Deposit': '{}k', 'Area': '{:.1f}m²'
                }),
                height=500,
                hide_index=True
            )
        else:
            st.write("No listings available.")

    st.divider()

    # (B) Outlier (Risky/Suspicious) Listings
    st.subheader("🚨 Warning! 'Outlier' Listings Matching Criteria")
    st.error("These listings match your criteria but are significantly cheaper or more expensive than market price. Please proceed with caution!")
    
    if not outliers.empty:
        col_out1, col_out2 = st.columns([2, 1])     
        with col_out1:
             st.write(f"**{len(outliers)}** suspicious listings were found.")
             
             outliers['color'] = [[255, 0, 0, 140] for _ in range(len(outliers))]
             outliers['size'] = 50
             
             st.map(outliers, color='color', size='size', zoom=14)
             
        with col_out2:
             st.write("📋 **Warning Listing List**")
             
            # Translate Outlier Reasons
             if 'outlier_reason' in outliers.columns:
                # 한글 키워드를 영어로 순차적으로 변환
                    outliers['Reason'] = (
                    outliers['outlier_reason']
                    .astype(str)
                    .str.replace('월세', 'Rent')
                    .str.replace('보증금', 'Deposit')
                    .str.replace('전용면적', 'Area')
                    .str.replace('_high', ' (High)')
                    .str.replace('_low', ' (Low)')
                )
             else:
                 outliers['Reason'] = "No Data"
             
             display_cols = ['월세', '보증금', '전용면적', '해당층', 'Reason']
             
             st.dataframe(
                outliers[display_cols].rename(columns={
                    '월세': 'Rent', '보증금': 'Deposit', '전용면적': 'Area', '해당층': 'Floor'
                }).style.format({
                    'Rent': '{:.1f}k', 'Deposit': '{}k', 'Area': '{:.1f}m²'
                }),
                height=500,
                hide_index=True
            )
    else:
        st.success("Fortunately, no 'Risky' listings were found for your criteria. Rest easy! 🛡️")
else:
    st.info("👈 Please set your conditions in the sidebar and click the 'Predict' button.")