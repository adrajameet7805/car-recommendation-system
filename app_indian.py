import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Indian Car Recommender", page_icon="🚗", layout="wide")

@st.cache_resource
def load_models():
    knn = joblib.load('models/knn.pkl')
    rf = joblib.load('models/rf.pkl')
    lr = joblib.load('models/lr.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoders = joblib.load('models/encoders.pkl')
    return knn, rf, lr, scaler, encoders

@st.cache_data
def load_data():
    df = pd.read_csv('indian-auto-mpg.csv')
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    df = df.rename(columns={
        'Manufacturer':    'brand',
        'Name':            'car_name',
        'Location':        'city',
        'Year':            'year',
        'Kilometers_Driven': 'mileage_km',
        'Fuel_Type':       'fuel_type',
        'Transmission':    'transmission',
        'Owner_Type':      'owner_type',
        'Engine CC':       'engine_cc',
        'Power':           'power_bhp',
        'Seats':           'seats',
        'Mileage Km/L':    'kmpl',
        'Price':           'price_lakh',
    })

    # Save raw for display
    df['price_inr'] = df['price_lakh'] * 100000
    df['price_raw'] = df['price_inr']
    df['mileage_raw'] = df['mileage_km']
    df['year_raw']    = df['year']

    # Clean
    df = df[df['year'] >= 2005].copy()
    df = df[(df['mileage_km'] >= 500) & (df['mileage_km'] <= 500000)].copy()
    df = df[(df['price_lakh'] >= 0.5) & (df['price_lakh'] <= 150)].copy()
    df = df[df['seats'] > 0].copy()
    df = df[df['power_bhp'] > 0].copy()
    df['kmpl'] = df['kmpl'].fillna(df['kmpl'].median())

    # Add tags
    avg_p = df.groupby('brand')['price_lakh'].transform('mean')
    df['price_vs_avg'] = df['price_lakh'] / avg_p
    df['deal_tag'] = df['price_vs_avg'].apply(
        lambda r: 'Good Deal'  if r <= 0.85 else ('Fair Price' if r <= 1.15 else 'Overpriced')
    )

    def calc_rating(row):
        score = 5.0
        if row['owner_type'] == 'Second': score -= 0.3
        if row['owner_type'] == 'Third':  score -= 0.7
        if row['owner_type'] == 'Fourth': score -= 1.2
        if row['mileage_km'] > 100000:    score -= 0.5
        if row['mileage_km'] > 150000:    score -= 0.5
        if row['year'] < 2010:            score -= 0.5
        return round(max(1.0, min(5.0, score + random.uniform(-0.2, 0.2))), 1)
        
    random.seed(42)
    df['rating'] = df.apply(calc_rating, axis=1)

    df = df.drop_duplicates(subset=['brand','car_name','year','mileage_km','price_lakh']).reset_index(drop=True)

    # Encode
    encode_map = {
        'fuel_type':    'fuel_enc',
        'transmission': 'transmission_enc',
        'owner_type':   'owner_enc',
        'city':         'city_enc',
        'brand':        'brand_enc',
        'car_name':     'name_enc',
    }
    
    encoders_dict = {}
    for raw_col, enc_col in encode_map.items():
        le = LabelEncoder()
        df[enc_col] = le.fit_transform(df[raw_col].astype(str))
        encoders_dict[raw_col] = le

    FEATURES = [
        'price_inr',    'mileage_km',
        'year',         'engine_cc',
        'power_bhp',    'seats',
        'kmpl',         'brand_enc',
        'name_enc',     'fuel_enc',
        'transmission_enc', 'owner_enc',
        'city_enc'
    ]
    df[FEATURES] = df[FEATURES].fillna(0)
    
    return df

def get_recommendations(user_prefs, df, knn, rf, top_n=10):
    FEATURES = [
        'price_inr',    'mileage_km',
        'year',         'engine_cc',
        'power_bhp',    'seats',
        'kmpl',         'brand_enc',
        'name_enc',     'fuel_enc',
        'transmission_enc', 'owner_enc',
        'city_enc'
    ]
    
    filtered = df.copy()

    # Price
    if user_prefs['max_budget'] > 0:
        filtered = filtered[filtered['price_lakh'] <= user_prefs['max_budget']]

    # KM
    if user_prefs['max_km'] > 0:
        filtered = filtered[filtered['mileage_km'] <= user_prefs['max_km']]

    # Year
    if user_prefs['min_year'] > 0:
        filtered = filtered[filtered['year'] >= user_prefs['min_year']]

    # Fuel
    multi_fuel = user_prefs.get('multi_fuel', [])
    if multi_fuel:
        filtered = filtered[filtered['fuel_type'].isin(multi_fuel)]
    elif user_prefs.get('fuel_type', 'Any') != 'Any':
        filtered = filtered[filtered['fuel_type'] == user_prefs['fuel_type']]

    # Brand
    multi_brand = user_prefs.get('multi_brand', [])
    if multi_brand:
        filtered = filtered[filtered['brand'].isin(multi_brand)]
    elif user_prefs.get('brand', 'Any') != 'Any':
        filtered = filtered[filtered['brand'] == user_prefs['brand']]

    # Transmission
    if user_prefs.get('transmission', 'Any') != 'Any':
        filtered = filtered[filtered['transmission'] == user_prefs['transmission']]

    # City
    if user_prefs.get('city', 'Any') != 'Any':
        filtered = filtered[filtered['city'] == user_prefs['city']]

    # Owner
    if user_prefs.get('owner', 'Any') != 'Any':
        filtered = filtered[filtered['owner_type'] == user_prefs['owner']]

    # Seats
    if user_prefs.get('seats', 'Any') != 'Any':
        filtered = filtered[filtered['seats'] == int(user_prefs['seats'])]
        
    # Mileage
    if user_prefs.get('min_kmpl', 0) > 0:
        filtered = filtered[filtered['kmpl'] >= user_prefs['min_kmpl']]

    if len(filtered) == 0:
        return pd.DataFrame()

    # Score
    X_filt = filtered[FEATURES].fillna(0)
    scores = rf.predict_proba(X_filt).max(axis=1) * 100
    filtered = filtered.copy()
    filtered['match_score'] = scores

    # Diversify
    filtered = filtered.sort_values('match_score', ascending=False)
    seen = {}
    results = []
    for _, row in filtered.iterrows():
        m = row['car_name']
        if seen.get(m, 0) < 2:
            results.append(row)
            seen[m] = seen.get(m, 0) + 1
        if len(results) == top_n:
            break
            
    return pd.DataFrame(results)

def main():
    st.title("🚗 Indian Car Recommendation Engine")
    st.markdown("Find your perfect used car using Machine Learning trained on real Indian data")

    try:
        knn, rf, lr, scaler, encoders = load_models()
        df = load_data()
    except Exception as e:
        st.error(f"Error loading models/data. Please run `python train_indian.py` first.\nDetails: {e}")
        return

    st.sidebar.header("🔍 Search Filters")

    # Budget
    budget_slider = st.sidebar.slider("Max Budget (₹ Lakhs)", min_value=0.5, max_value=150.0, value=15.0, step=0.5)
    st.sidebar.markdown("**Or type exact budget:**")
    budget_input = st.sidebar.number_input("Enter exact budget", min_value=0.5, max_value=500.0, value=float(budget_slider), step=0.5, label_visibility="collapsed")
    final_budget = max(budget_slider, budget_input)

    # Year
    min_year = st.sidebar.slider("Minimum Year", min_value=2005, max_value=2019, value=2012, step=1)

    # KM
    km_slider = st.sidebar.slider("Max Kilometers Driven", min_value=0, max_value=300000, value=80000, step=5000)
    st.sidebar.markdown("**Or type exact km:**")
    km_input = st.sidebar.number_input("Enter exact km", min_value=0, max_value=1000000, value=int(km_slider), step=5000, label_visibility="collapsed")
    final_km = max(km_slider, km_input)

    # Fuel
    fuel_options = ['Any', 'Petrol', 'Diesel', 'CNG', 'LPG']
    fuel = st.sidebar.selectbox("Fuel Type", fuel_options)
    multi_fuel = st.sidebar.multiselect("Select Multiple Fuel Types", options=['Petrol', 'Diesel', 'CNG', 'LPG'], default=[])

    # Trans
    trans_options = ['Any', 'Manual', 'Automatic']
    transmission = st.sidebar.selectbox("Transmission", trans_options)

    # Owner
    owner_options = ['Any', 'First', 'Second', 'Third']
    owner = st.sidebar.selectbox("Owner Type", owner_options)

    # Seats
    seats_options = ['Any', '2', '4', '5', '6', '7', '8']
    seats = st.sidebar.selectbox("Seats", seats_options)

    # Brand
    brand_list = sorted(df['brand'].astype(str).unique().tolist())
    brand_options = ['Any'] + brand_list
    brand = st.sidebar.selectbox("Brand", brand_options)
    multi_brand = st.sidebar.multiselect("Select Multiple Brands", options=brand_list, default=[])

    # City
    city_options = ['Any', 'Mumbai', 'Hyderabad', 'Pune', 'Delhi', 'Bangalore', 'Kochi', 'Chennai', 'Kolkata', 'Jaipur', 'Ahmedabad']
    city = st.sidebar.selectbox("City", city_options)

    # Mileage
    kmpl_slider = st.sidebar.slider("Min Mileage (km/L)", min_value=5.0, max_value=35.0, value=12.0, step=1.0)
    st.sidebar.markdown("**Or type exact value:**")
    kmpl_input = st.sidebar.number_input("Enter exact km/L", min_value=0.0, max_value=50.0, value=float(kmpl_slider), step=1.0, label_visibility="collapsed")
    final_kmpl = max(kmpl_slider, kmpl_input)

    user_prefs = {
        'max_budget': final_budget,
        'min_year': min_year,
        'max_km': final_km,
        'fuel_type': fuel,
        'multi_fuel': multi_fuel,
        'transmission': transmission,
        'owner': owner,
        'seats': seats,
        'brand': brand,
        'multi_brand': multi_brand,
        'city': city,
        'min_kmpl': final_kmpl
    }

    if st.sidebar.button("🔍 Find My Car", use_container_width=True):
        with st.spinner("Analyzing cars..."):
            recs = get_recommendations(user_prefs, df, knn, rf, top_n=10)
            
            if len(recs) == 0:
                st.markdown("### 😔 No cars found")
                st.warning("We couldn't find any matches. Try adjusting your filters.")
                return
                
            st.markdown("### ✨ Top Recommendations")
            
            for idx, row in recs.iterrows():
                deal_color = "red"
                deal_icon = "🔴"
                if row['deal_tag'] == "Good Deal":
                    deal_color = "green"
                    deal_icon = "✅"
                elif row['deal_tag'] == "Fair Price":
                    deal_color = "#FFC300"
                    deal_icon = "🟡"
                
                # Format numbers
                km_fmt = f"{int(row['mileage_km']):,}"
                
                card_html = (
                    f'<div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 20px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">'
                    f'<div style="display: flex; justify-content: space-between;">'
                    f'<div>'
                    f'<h4 style="margin-top:0; margin-bottom: 10px; color: #2c3e50;">🚗 {row["car_name"]} ({int(row["year"])})</h4>'
                    f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">'
                    f'<div><b>🏢 Brand:</b> {row["brand"]}</div>'
                    f'<div><b>💰 Price:</b> ₹{row["price_lakh"]:.2f} Lakhs</div>'
                    f'<div><b>📍 KM Driven:</b> {km_fmt} km</div>'
                    f'<div><b>⛽ Fuel:</b> {row["fuel_type"]}</div>'
                    f'<div><b>⚙️ Transmission:</b> {row["transmission"]}</div>'
                    f'<div><b>🔧 Engine:</b> {int(row["engine_cc"])} cc</div>'
                    f'<div><b>⚡ Power:</b> {int(row["power_bhp"])} bhp</div>'
                    f'<div><b>💺 Seats:</b> {int(row["seats"])}</div>'
                    f'<div><b>👤 Owner:</b> {row["owner_type"]}</div>'
                    f'<div><b>⛽ Mileage:</b> {row["kmpl"]:.1f} km/L</div>'
                    f'<div><b>🏙️ City:</b> {row["city"]}</div>'
                    f'<div><b>⭐ Rating:</b> {row["rating"]:.1f}/5.0</div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="text-align: right; min-width: 150px; background-color: #f8f9fa; padding: 15px; border-radius: 8px;">'
                    f'<div style="font-size: 24px; font-weight: bold; color: {deal_color}; text-align: center;">{deal_icon}<br/><span style="font-size: 16px;">{row["deal_tag"]}</span></div>'
                    f'<div style="margin-top: 15px; text-align: center;">'
                    f'<div style="font-size: 12px; color: #666;">Match Score</div>'
                    f'<div style="font-size: 22px; font-weight: bold; color: #4CAF50;">{row["match_score"]:.1f}%</div>'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
