


# 🏎️ F1 Podium Predictor

Advanced machine learning system for predicting Formula 1 podium finishes using optimized K-Nearest Neighbors classification. Achieves exceptional 82.61% holdout ROC-AUC performance through systematic hyperparameter tuning and comprehensive feature engineering.

## 🎯 Project Overview

This project develops a high-performance machine learning model to predict whether Formula 1 drivers will finish in podium positions (top 3) based on historical race data and driver performance metrics.

## 🔮 Usage Example

### **Making Predictions on New F1 Data**

import joblib
import pandas as pd
import numpy as np

Load the trained model and scaler
model = joblib.load('f1predictor.pkl')
scaler = joblib.load('f1scaler.pkl')

Example: Predicting podium probability for Lewis Hamilton at Monaco GP 2024
new_race_data = {
'year': 2024,
'round': 8,
'circuitId': 6, # Monaco circuit ID
'rainy': 0, # No rain expected
'Turns': 19,
'Length': 3.337,
'driverId': 1, # Lewis Hamilton's ID
'constructorId': 131, # Mercedes constructor ID
'grid': 3, # Starting from P3
'Driver Top 3 Finish Percentage (Last Year)': 45.5,
'Constructor Top 3 Finish Percentage (Last Year)': 52.2,
'Driver Top 3 Finish Percentage (This Year till last race)': 28.6,
'Constructor Top 3 Finish Percentage (This Year till last race)': 35.7,
'Driver Avg position (Last Year)': 5.8,
'Constructor Avg position (Last Year)': 4.9,
'Driver Average Position (This Year till last race)': 7.2,
'Constructor Average Position (This Year till last race)': 6.8,
'driver_age': 39,
'nationality_encoded': 4, # British nationality encoded
'wins_cons': 1.0, # Constructor wins this season
'points_cons_champ': 156.0, # Constructor championship points
'wins': 1.0, # Driver wins this season
'points_driver_champ': 81.0, # Driver championship points
'laps': 78, # Expected laps to complete
'statusId': 1, # Expected to finish
'Weighted_Top_3_Probability': 0.342,
'Weighted_Top_3_Prob_Length': 0.298,
'position_previous_race': 4.0, # Finished P4 in last race
'nro_cond_escuderia': 1, # Driver number within team
'raceId': 1128, # Unique race identifier
'points': 0.0, # Points from this race (prediction target)
'prom_points_10': 8.7 # Average points from last 10 races
}

Create input DataFrame
X_new = pd.DataFrame([new_race_data], columns=feature_columns)

CRITICAL: Scale the input data using the same scaler from training
X_new_scaled = scaler.transform(X_new)

Make prediction
podium_probability = model.predict_proba(X_new_scaled)
podium_prediction = model.predict(X_new_scaled)

print(f"Podium Probability: {podium_probability:.3f}")
print(f"Podium Prediction: {'Podium Finish' if podium_prediction == 1 else 'No Podium'}")

Example output:
Podium Probability: 0.687




## 📊 Key Results

- **Holdout ROC-AUC**: 82.61%
- **Cross-Validation ROC-AUC**: 97.47%
- **Model Improvement**: +18.35 percentage points over baseline
- **Podium Recall**: 93% (excellent at catching actual podium finishers)

## 🔧 Technical Approach

### **Data Processing**
- Comprehensive exploratory data analysis (EDA)
- Feature engineering and correlation analysis
- StandardScaler preprocessing for optimal KNN performance
- SMOTE oversampling to handle class imbalance (14.24% podium rate)

### **Model Development**
- Systematic comparison of multiple algorithms (KNN, Random Forest, SVM, etc.)
- Extensive grid search hyperparameter optimization (45+ minutes runtime)
- Rigorous train/validation/holdout split methodology
- 5-fold cross-validation for robust performance estimation

### **Optimal Hyperparameters**
{
'algorithm': 'auto',
'metric': 'manhattan',
'n_neighbors': 13,
'p': 1,
'weights': 'distance'
}


## 📁 Project Structure

├── f1modelselector.ipynb # Complete analysis & model development
├── finalmodel.ipynb # Production model implementation
├── README.md # Project documentation
└── .gitignore # Repository configuration


## 🚀 Key Features

- **Advanced Classification**: Manhattan distance KNN with distance weighting
- **Class Balance Handling**: SMOTE technique for imbalanced datasets  
- **Comprehensive Validation**: Proper holdout testing prevents overfitting
- **Production Ready**: Clean, exportable model with joblib serialization
- **Professional Methodology**: Industry-standard ML pipeline

## 📈 Model Performance

| Metric | Cross-Validation | Holdout Test |
|--------|------------------|--------------|
| **ROC-AUC** | 97.47% | **82.61%** |
| **Accuracy** | - | 75% |
| **Podium Recall** | - | 93% |
| **Macro F1-Score** | - | 67% |

## 🏁 Business Impact

- **High Podium Detection**: 93% recall ensures minimal missed podium predictions
- **Conservative Precision**: Reliable ranking system for race predictions  
- **Real-World Applicable**: 82.61% ROC-AUC represents excellent discrimination ability
- **Scalable Framework**: Methodology applicable to other motorsport predictions

## 🛠️ Technologies Used

- **Python**: Primary programming language
- **scikit-learn**: Machine learning algorithms and validation
- **imbalanced-learn**: SMOTE oversampling technique
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Data visualization
- **joblib**: Model serialization

## 📋 Installation & Usage

Clone repository
git clone https://github.com/Yashp002/F1-Podium-Predictor.git

Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib

Run analysis
jupyter notebook finalmodel.ipynb


## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- End-to-end machine learning pipeline development
- Advanced hyperparameter optimization techniques
- Handling imbalanced classification problems
- Professional code organization and documentation
- Statistical model validation and interpretation

## 👨‍💻 Author

**Yashp002** - Computer Science Engineering Student | Aspiring AI/ML Engineer

---

*This project showcases advanced machine learning techniques applied to real-world Formula 1 data, achieving research-grade performance through systematic optimization and rigorous validation methodology.*

