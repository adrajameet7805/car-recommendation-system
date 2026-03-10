<<<<<<< HEAD
# 🚗 Indian Car Recommendation Engine

## 📋 Project Overview
The **Indian Car Recommendation Engine** is an end-to-end Machine Learning web application designed to help users find the perfect used car in the Indian market. By leveraging a comprehensive real-world dataset, the application provides personalized car recommendations, evaluates market prices (Deal Tags), and scores vehicles based on user preferences. It features a clean, interactive dashboard built with Streamlit.

## 🎯 Problem Statement
Navigating the used car market is overwhelming. Buyers often struggle to determine if a requested price is fair, evaluate the condition of a vehicle based on mileage and age, or find cars that accurately match their strict constraints (budget, fuel type, transmission, seating, etc.). This project solves this by using Machine Learning to automatically cluster similar vehicles, rank them by a predicted "Match Score", and explicitly tag them as a "Good Deal", "Fair Price", or "Overpriced" based on historical market averages.

## 💻 Technologies Used
* **Programming Language:** Python 3.8+
* **Frontend Framework:** Streamlit
* **Machine Learning:** Scikit-Learn
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Plotly
* **Model Serialization:** Joblib

## 🏗️ System Architecture
The system is divided into two primary pipelines:

1. **Backend Training Pipeline (`train_indian.py`)** 
   * Ingests the raw CSV dataset.
   * Cleans data (removes outliers, handles missing values).
   * Generates heuristic features (Deal Tags, Ratings).
   * Encodes categorical variables (Label Encoding) and scales continuous numerical features (MinMaxScaler).
   * Trains 4 distinct ML models (KNN, Random Forest, Linear Regression, Logistic Regression).
   * Serializes the models, scalers, encoders, and cleaned dataset into the `models/` directory.

2. **Frontend Inference Pipeline (`app_indian.py`)**
   * Bootstraps the Streamlit web server.
   * Loads serialized artifacts into memory (`@st.cache_resource`).
   * Captures user inputs from the visual sidebar.
   * Filters the dataset heuristically based on hard constraints.
   * Dynamically encodes the filtered subset and feeds it into the Random Forest model to generate `match_scores`.
   * Enforces clustering diversity (maximum of 2 variations per car model).
   * Renders the final recommendations as aesthetic HTML cards.

## 🔄 Project Workflow (Step-by-Step)
1. **Data Ingestion:** The dataset is loaded and cleaned. Cars older than 2005, with unrealistic prices (<0.5 Lakhs or >150 Lakhs), or impossible mileages are dropped.
2. **Feature Engineering:** Missing values (like kmpl) are imputed using medians. A `deal_tag` is assigned by comparing the car's price to its model's average market price. A `rating` out of 5.0 is dynamically generated based on vehicle age, mileage, and owner history.
3. **Model Training:** `train_indian.py` fits the models and saves them as `.pkl` files.
4. **App Initialization:** The user runs `streamlit run app_indian.py`.
5. **User Interaction:** The user adjusts sliders and dropdowns (Budget, Kilometers Driven, Fuel Type, Seats, etc.).
6. **Filtering:** The application strictly applies these constraints via Pandas masks.
7. **Scoring & Sorting:** The remaining viable cars are passed through the trained Random Forest pipeline (`predict_proba`) to generate a percentage Match Score.
8. **Results Rendering:** The top 10 most relevant, diverse cars are rendered in colorful UI cards.

## ✨ Features of the System
* **10+ Granular Filters:** Exact Budget inputs, Multi-select Fuel Types, Transmissions, Owner Types, Seats, Brands, and specific Indian Cities.
* **Smart UI Layout:** Beautifully formatted HTML recommendation cards with emojis and clear visual hierarchy.
* **Deal Evaluator:** Highlights cars in Green (`✅ Good Deal`), Yellow (`🟡 Fair Price`), or Red (`🔴 Overpriced`).
* **Diversity Engine:** Prevents the UI from being flooded by the exact same car model by capping identical models to a maximum of two.
* **Localized Currency:** Displays pricing in explicit Indian Lakhs (₹X.XX Lakhs).
* **Caching:** High-performance, instant reloading via Streamlit caching constraints.

## 🧠 Machine Learning Model Explanation
The project utilizes a distinct ensemble of models to achieve its results:
1. **Random Forest Classifier (Primary):** Used in the live application to calculate the `match_score`. It utilizes `.predict_proba()` to determine the confidence level that a specific car is highly recommendable (Rating >= 4.0) based on all unified features.
2. **K-Nearest Neighbors (KNN):** Trained on scaled numerical features and encoded categoricals to map the exact spatial similarity between different vehicles.
3. **Linear Regression:** Trained to predict the exact numerical fair-market price (`price_lakh`) based on existing depreciation curves (mileage, year, owner type).
4. **Logistic Regression:** A swift binary classification fallback to determine a simple "Yes/No" recommendation boundary.

## 📊 Dataset Explanation
* **Source:** `indian-auto-mpg.csv` (Real-world Indian used car market data).
* **Size:** 5,975 instances, 14 Features.
* **Columns:** `Name`, `Manufacturer`, `Location` (Mumbai, Hyderabad, Delhi, etc.), `Year`, `Kilometers_Driven`, `Fuel_Type`, `Transmission`, `Owner_Type`, `Engine CC`, `Power`, `Seats`, `Mileage Km/L`, `Price (in Lakhs ₹)`.
* **Preprocessing:** `Unnamed: 0` artifacts deleted. Column names normalized for Python syntax. Outliers surgically removed. 

## 🚀 How the Application Works
1. Navigate to the project directory in your terminal.
2. Run the environment training pipeline to parse the dataset:
   ```bash
   python train_indian.py
   ```
3. Boot the Streamlit server:
   ```bash
   streamlit run app_indian.py
   ```
4. Adjust the sidebar parameters in your browser at `http://localhost:8501` to view your personalized recommendations.

## 🌍 Real-World Applications
* **Dealership Platforms:** Can be integrated into existing dealership frontends (like Cars24 or Spinny) to suggest alternatives to users.
* **Consumer Empowerment:** Allows individual buyers to gut-check dealership prices against the ML-predicted "Fair Price" market average.
* **Financial Services:** Banks assessing auto-loans can use the underlying Linear Regression model to evaluate the asset's true collateral value.

## 💡 Future Improvements & Optimization Suggestions
* **Hyperparameter Tuning:** Implement `GridSearchCV` or `Optuna` during training to mathematically optimize the Random Forest depth and KNN neighbor count.
* **API Decoupling:** Separate the ML inference into a FastAPI/Flask backend microservice, allowing the Streamlit frontend to make external HTTP requests. This improves horizontal scaling.
* **Advanced NLP:** Use Natural Language Processing to parse descriptive text blocks (if available in future datasets) for sentiment analysis regarding vehicle condition.
* **Image Integration:** Connect the dataset to an external API (like Google Custom Search) to dynamically pull and display actual thumbnails of the specific car models evaluated. 
* **Additional Visualizations:** Introduce a scatter plot visualization comparing the user's filtered mileage vs. price array natively in the dashboard.
=======
# car-recommendation-system
🚗 Indian Car Recommendation Engine is a Machine Learning web app built with Python and Streamlit that helps users find the best used cars in India. It analyzes market data, evaluates price fairness, and recommends vehicles using ML models like Random Forest, KNN, and Regression based on user preferences.
>>>>>>> 59d3cf3f37f3b68fc9aeebf72285ad8f58f1b7d8
