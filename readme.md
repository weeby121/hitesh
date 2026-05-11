# 🫀 Heart Disease Prediction Using Machine Learning

## 📌 Overview
This project aims to predict the likelihood of heart disease in patients based on their clinical parameters (age, sex, cholesterol, blood pressure, etc.) using Machine Learning techniques. By building a classification model, this tool assists in early diagnosis and risk assessment.

*   **Goal:** Binary Classification (Presence/Absence of Heart Disease)
*   **Best Model:** [E.g., RandomForest/XGBoost]
*   **Accuracy:** [E.g., 91%]


## 📊 Dataset Description
The dataset used is commonly the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), which contains 14 key features:
*   **age**: Age in years
*   **sex**: 1 = male; 0 = female
*   **cp**: Chest pain type (4 types)
*   **trestbps**: Resting blood pressure (in mm Hg)
*   **chol**: Serum cholesterol in mg/dL
*   **fbs**: Fasting blood sugar > 120 mg/dL (1=true, 0=false)
*   **restecg**: Resting electrocardiographic results
*   **thalach**: Maximum heart rate achieved
*   **exang**: Exercise-induced angina (1=yes, 0=no)
*   **target**: Diagnosis (1 = disease, 0 = no disease)

## 🛠️ Technologies Used
*   **Python 3.9+**
*   **Pandas & NumPy** – Data Manipulation
*   **Matplotlib & Seaborn** – Data Visualization
*   **Scikit-Learn** – Machine Learning Algorithms
*   **Joblib** – Model Serialization
*   **Streamlit** – Web Application Deployment

## ⚙️ Steps Performed
1.  **Data Cleaning:** Handled missing values and mapped categorical features.
2.  **Exploratory Data Analysis (EDA):** Visualized correlations between heart disease and patient attributes.
3.  **Feature Scaling:** Applied `StandardScaler` to numerical features for improved model performance.
4.  **Model Training:** Evaluated algorithms including Logistic Regression, Random Forest, and SVM.
5.  **Hyperparameter Tuning:** Optimized the best-performing model (e.g., using GridSearchCV).

## 🚀 How to Run
1.  Clone the repository:
    ```bash
    git clone https://github.com
    ```
2.  Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the web application:
    ```bash
    streamlit run app.py
    ```

## 📈 Results
The Random Forest Classifier outperformed other models with the following results:
*   **Accuracy:** 91%
*   **Precision:** 0.89
*   **Recall:** 0.92
