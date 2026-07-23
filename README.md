# Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn using customer demographics, account information, and service usage patterns.

## Features

* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Machine Learning Pipeline using Scikit-Learn
* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier
* Feature Importance Analysis
* Interactive Streamlit Web Application
* Model Serialization using Joblib

---

## Dataset

**Dataset:** Telco Customer Churn Dataset

The dataset contains customer information such as:

* Gender
* Senior Citizen Status
* Partner & Dependents
* Contract Type
* Internet Service
* Monthly Charges
* Total Charges
* Tenure
* Payment Method
* Churn Status

Target Variable:

```text
Churn
0 -> Customer Stays
1 -> Customer Leaves
```

---

## Exploratory Data Analysis

Key findings from EDA:

### Contract Type

| Contract       | Churn Rate |
| -------------- | ---------: |
| Month-to-month |      42.7% |
| One Year       |      11.3% |
| Two Year       |       2.8% |

Month-to-month customers were significantly more likely to churn.

### Tenure

Customers who churned generally had much lower tenure than retained customers.

### Monthly Charges

Customers with higher monthly charges showed a higher tendency to churn.

---

## Data Preprocessing

### Numerical Features

* SeniorCitizen
* tenure
* MonthlyCharges
* TotalCharges

Processing:

* Missing Value Imputation
* Standard Scaling

### Categorical Features

* Gender
* Partner
* Dependents
* Internet Service
* Contract
* Payment Method
* Other service-related attributes

Processing:

* Missing Value Imputation
* One-Hot Encoding

Implemented using:

```python
ColumnTransformer
Pipeline
```

---

##  Models Trained

### Logistic Regression

* Baseline Model
* Fast and interpretable

### Random Forest Classifier

* Ensemble Learning
* Bagging-based approach

### XGBoost Classifier

* Gradient Boosting Framework
* Handles complex feature interactions

---

## Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   79.53% |    63.75% | 53.30% |   58.06% |
| Random Forest       |   77.68% |    60.71% | 45.45% |   51.99% |
| XGBoost             |   79.62% |    64.65% | 51.52% |   57.34% |

### Best Performing Model

Logistic Regression achieved the highest F1 Score and Recall, making it the preferred model for deployment.

---

## Feature Importance (XGBoost)

Top churn-driving features:

1. Contract_Month-to-month
2. InternetService_Fiber optic
3. TechSupport_No
4. Contract_Two year
5. OnlineSecurity_No
6. Tenure
7. MonthlyCharges

These insights aligned closely with findings from the exploratory data analysis.

---

##  Streamlit Application

The project includes a Streamlit-based web interface where users can:

* Enter customer details
* Predict churn probability
* Identify customers at risk of leaving

Run locally:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Dataset.csv
│
├── models/
│   └── churn_pipeline.pkl
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── src/
│   └── train.py
│
└── screenshots/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
```

Move into the project directory:

```bash
cd customer-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧪 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Streamlit
* Joblib
* Matplotlib

---

## 🎯 Future Improvements

* Hyperparameter Tuning
* ROC-AUC Analysis
* SHAP Explainability
* Cross Validation
* Docker Deployment
* Cloud Deployment (Render / Streamlit Cloud)

---

## 👨‍💻 Author

**Arpit Shirbhate**
---
If you found this project useful, consider giving it a ⭐ on GitHub.
