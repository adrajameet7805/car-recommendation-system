import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import joblib
import os
import random

def train_models():
    print("Loading indian-auto-mpg.csv...")
    if not os.path.exists('indian-auto-mpg.csv'):
        print("Error: indian-auto-mpg.csv not found!")
        return
        
    df = pd.read_csv('indian-auto-mpg.csv')
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
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

    # Convert price to INR for display
    df['price_inr'] = df['price_lakh'] * 100000
    df['price_raw'] = df['price_inr']
    df['mileage_raw'] = df['mileage_km']
    df['year_raw']    = df['year']

    # STEP 2 — DATA CLEANING
    print("Cleaning data...")
    # Remove too old cars
    df = df[df['year'] >= 2005].copy()

    # Remove unrealistic mileage
    df = df[(df['mileage_km'] >= 500) & (df['mileage_km'] <= 500000)].copy()

    # Remove unrealistic prices
    df = df[(df['price_lakh'] >= 0.5) & (df['price_lakh'] <= 150)].copy()

    # Remove 0 seats
    df = df[df['seats'] > 0].copy()

    # Fix power outliers
    df = df[df['power_bhp'] > 0].copy()

    # Fill missing kmpl
    df['kmpl'] = df['kmpl'].fillna(df['kmpl'].median())

    # Add deal tag vs model average price
    avg_p = df.groupby('brand')['price_lakh'].transform('mean')
    df['price_vs_avg'] = df['price_lakh'] / avg_p
    df['deal_tag'] = df['price_vs_avg'].apply(
        lambda r: 'Good Deal'  if r <= 0.85 else ('Fair Price' if r <= 1.15 else 'Overpriced')
    )

    # Add rating based on owner, km, year
    random.seed(42)
    def calc_rating(row):
        score = 5.0
        if row['owner_type'] == 'Second': score -= 0.3
        if row['owner_type'] == 'Third':  score -= 0.7
        if row['owner_type'] == 'Fourth': score -= 1.2
        if row['mileage_km'] > 100000:    score -= 0.5
        if row['mileage_km'] > 150000:    score -= 0.5
        if row['year'] < 2010:            score -= 0.5
        return round(max(1.0, min(5.0, score + random.uniform(-0.2, 0.2))), 1)

    df['rating'] = df.apply(calc_rating, axis=1)

    # Remove duplicates
    df = df.drop_duplicates(subset=['brand','car_name','year','mileage_km','price_lakh']).reset_index(drop=True)

    # STEP 3 — FEATURE ENGINEERING
    print("Engineering features...")
    encode_map = {
        'fuel_type':    'fuel_enc',
        'transmission': 'transmission_enc',
        'owner_type':   'owner_enc',
        'city':         'city_enc',
        'brand':        'brand_enc',
        'car_name':     'name_enc',
    }
    encoders = {}
    for raw_col, enc_col in encode_map.items():
        le = LabelEncoder()
        df[enc_col] = le.fit_transform(df[raw_col].astype(str))
        encoders[raw_col] = le

    # Final ML feature list
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
    
    # Save the cleaned dataset for the app
    os.makedirs('models', exist_ok=True)
    df.to_csv('models/df_cleaned_indian.csv', index=False)

    # STEP 4 — TRAIN ML MODELS
    print("Training ML models...")
    X = df[FEATURES]
    y_brand = df['brand_enc']
    y_recommend = (df['rating'] >= 4.0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y_brand, test_size=0.2, random_state=42)

    # Scale
    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # 1. KNN
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train_sc, y_train)
    print("KNN Accuracy:", accuracy_score(y_test, knn.predict(X_test_sc)))

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))

    # 3. Linear Regression (price prediction)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print("Linear Regression R2:", r2_score(y_test, lr.predict(X_test)))

    # 4. Logistic Regression (recommend yes/no)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y_recommend, test_size=0.2, random_state=42)
    log_reg = LogisticRegression(max_iter=500)
    log_reg.fit(Xr_tr, yr_tr)
    print("Logistic Regression Accuracy:", accuracy_score(yr_te, log_reg.predict(Xr_te)))

    # 5. Cross validation
    cv = cross_val_score(rf, X, y_brand, cv=5)
    print(f"RF CrossVal: {cv.mean():.3f} ± {cv.std():.3f}")

    # Save models
    joblib.dump(knn,     'models/knn.pkl')
    joblib.dump(rf,      'models/rf.pkl')
    joblib.dump(lr,      'models/lr.pkl')
    joblib.dump(scaler,  'models/scaler.pkl')
    joblib.dump(encoders,'models/encoders.pkl')
    print("✅ All models saved!")

if __name__ == "__main__":
    train_models()
