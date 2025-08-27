


# 🏎️ F1 Podium Predictor

Advanced machine learning system for predicting Formula 1 podium finishes using optimized K-Nearest Neighbors classification. Achieves exceptional 82.61% holdout ROC-AUC performance through systematic hyperparameter tuning and comprehensive feature engineering.

## 🎯 Project Overview

This project develops a high-performance machine learning model to predict whether Formula 1 drivers will finish in podium positions (top 3) based on historical race data and driver performance metrics.

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

