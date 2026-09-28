<div align="center">

# ChurnIQ

### Customer Churn Prediction & Risk Intelligence

Predict churn probability for individual customers or analyze an entire customer dataset with an Artificial Neural Network.

<p>
  <img src="https://img.shields.io/badge/Model-ANN%20%7C%20MLP-111827?style=for-the-badge" alt="ANN">
  <img src="https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Accuracy-85.95%25-16a34a?style=for-the-badge" alt="Accuracy">
  <img src="https://img.shields.io/badge/License-AGPL--3.0-blue?style=for-the-badge" alt="AGPL-3.0">
</p>

**Single prediction · Bulk CSV analysis · Risk ranking · REST API**

</div>

---

## What is ChurnIQ?

**ChurnIQ** is a full-stack customer churn prediction application that turns an ANN/MLP model into an interactive analytics tool.

It supports two workflows:

| Workflow | What it does |
|---|---|
| 👤 **Single Prediction** | Enter one customer profile and receive a churn probability and risk level |
| 📊 **Dataset Analysis** | Upload a CSV, score every valid customer, rank the highest-risk customers and export results |

The goal is simple: **turn customer data into a clear risk signal that can be explored through a web interface.**

## ✨ Features

- Individual churn prediction
- CSV bulk prediction
- Automatic dataset validation
- Churn probability for every valid customer
- Low / Medium / High risk bands
- Highest-risk customer ranking
- Dataset summary statistics
- Downloadable prediction CSV
- FastAPI REST API
- Responsive web interface
- ANN/MLP model with **16 → 8 → 1** architecture
- Train/test preprocessing without test-set leakage
- Model metrics displayed in the application

---

## How it works

~~~text
                         CHURNIQ

       ┌─────────────────────────────────────┐
       │          Customer Data              │
       └──────────────────┬──────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
       Single Customer          CSV Dataset
              │                       │
              └───────────┬───────────┘
                          ▼
                  Input Validation
                          │
                          ▼
                Feature Preprocessing
                          │
                          ▼
                    ANN / MLP
                    16 → 8 → 1
                          │
                          ▼
                 Churn Probability
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
            Risk Level        Prediction
          Low / Medium /       Churn / Stay
               High
~~~

The runnable application uses scikit-learn MLPClassifier. The TensorFlow/Keras implementation of the same architecture is included in backend/train_keras.py.

## 📊 Model

### Architecture

~~~text
Input Features
      │
      ▼
┌──────────────────┐
│ Dense: 16        │
│ Activation: ReLU │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Dense: 8         │
│ Activation: ReLU │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Dense: 1         │
│ Binary output    │
└──────────────────┘
~~~

### Preprocessing

1. Split the dataset into training and test sets.
2. Fit preprocessing only on training data.
3. One-hot encode categorical features.
4. Standardize numerical features.
5. Train the ANN/MLP.
6. Evaluate against the untouched test set.

This prevents test-set information from leaking into preprocessing.

## Input features

| Feature | Type | Description |
|:--|:--:|:--|
| CreditScore | Numeric | Customer credit score |
| Geography | Categorical | France, Germany or Spain |
| Gender | Categorical | Male or Female |
| Age | Numeric | Customer age |
| Tenure | Numeric | Years with the institution |
| Balance | Numeric | Account balance |
| NumOfProducts | Numeric | Number of products held |
| HasCrCard | Binary | Credit-card ownership |
| IsActiveMember | Binary | Customer activity status |
| EstimatedSalary | Numeric | Estimated salary |

Identifier fields such as CustomerId, Surname and RowNumber can be present in uploaded files for identification, but they are not model inputs.

## Understanding feature changes

The ANN learns relationships between the input variables rather than applying fixed rules. A change to Age, Balance, Geography, activity, product count or another input can therefore change the predicted probability depending on the complete customer profile.

- **Age** contributes to the learned churn pattern across customer profiles.
- **Geography** allows the model to learn regional differences in the training data.
- **NumOfProducts** captures product relationship depth.
- **IsActiveMember** provides an engagement signal.
- **Balance** provides account-level information.
- **Tenure** captures relationship duration.
- **CreditScore** provides a credit-profile signal.
- **EstimatedSalary** provides an income-related signal.
- **HasCrCard** and **Gender** provide additional binary/categorical information.

> A change in predicted probability is **not proof of causation**. For detailed feature attribution, SHAP or another explainability method can be added.

---

# 📊 Dataset Analysis

Upload a customer CSV and analyze the entire dataset at once.

### Required columns

~~~text
CreditScore
Geography
Gender
Age
Tenure
Balance
NumOfProducts
HasCrCard
IsActiveMember
EstimatedSalary
~~~

### Optional identifier columns

~~~text
CustomerId
Surname
RowNumber
~~~

### Processing pipeline

~~~text
CSV Upload → Validation → Valid / Invalid Rows → Prediction → Risk → Ranking → Export
~~~

ChurnIQ reports total rows, valid rows, invalid rows, predicted churn, risk distribution and average churn probability. The interface also shows the highest-risk customers first.

### Output

The exported prediction file adds:

~~~text
ChurnProbability
Prediction
Risk
~~~

| Customer | Probability | Risk | Prediction |
|:--|--:|:--:|:--|
| Customer A | 87.40% | High | Churn |
| Customer B | 72.15% | High | Churn |
| Customer C | 41.80% | Medium | Stay |
| Customer D | 8.62% | Low | Stay |

> The probability is a model estimate, not a guarantee that a customer will churn.

## 📈 Model performance

Current held-out test-set results:

| Metric | Result |
|:--|--:|
| **Accuracy** | **85.95%** |
| Churn Precision | 76.92% |
| Churn Recall | 44.23% |
| Churn F1 | 56.16% |
| Training Samples | 8,000 |
| Test Samples | 2,000 |

### Confusion matrix

~~~text
                    Predicted
                 Non-churn   Churn
Actual Non          1539       54
Actual Churn         227      180
~~~

The default binary classification threshold is **50%**. Because churn recall is lower than overall accuracy, probability and risk should be considered alongside the binary prediction.

## 🔌 API

| Method | Endpoint | Purpose |
|:--:|:--|:--|
| GET | /api/health | API and model health |
| GET | /api/model-info | Model metrics and metadata |
| POST | /api/predict | Predict one customer |
| POST | /api/predict-csv | Analyze a CSV dataset |
| POST | /api/predict-csv/download | Download scored CSV |

### Example request

~~~json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 35,
  "Tenure": 5,
  "Balance": 75000,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 100000
}
~~~

---

# 🚀 Quick Start

### Requirements

- Python 3.10+
- pip
- Git

### 1. Clone

~~~bash
git clone https://github.com/dh6xxn/churniq.git
cd churniq
~~~

### 2. Create a virtual environment

**Windows**

~~~powershell
python -m venv .venv
.venv\Scripts\activate
~~~

**macOS / Linux**

~~~bash
python -m venv .venv
source .venv/bin/activate
~~~

### 3. Install dependencies

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

### 4. Start

~~~bash
python -m uvicorn backend.main:app --reload
~~~

Open **http://127.0.0.1:8000**.

> **Windows note:** If `uvicorn` is reported as “not recognized”, do not run `uvicorn ...` directly. Run `python -m uvicorn backend.main:app --reload` from the activated virtual environment. If you installed dependencies without activating `.venv`, activate it first with `.venv\\Scripts\\activate`, then run `python -m pip install -r requirements.txt`.


### Retrain the model

Place Churn_Modelling.csv in data/ and run:

~~~bash
python backend/train_model.py
~~~

The raw dataset is intentionally not tracked by Git.

## 🗂️ Project structure

~~~text
churniq/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── train_model.py          # scikit-learn ANN/MLP training
│   ├── train_keras.py          # TensorFlow/Keras implementation
│   └── model/
│       └── metrics.json        # Evaluation metrics
├── frontend/
│   ├── index.html              # Application UI
│   ├── app.js                  # Frontend logic
│   └── styles.css              # UI styling
├── data/
│   └── README.md
├── LICENSE
├── README.md
└── requirements.txt
~~~

## 🔐 Data & privacy

ChurnIQ is designed to run locally. Uploaded CSV data is processed by the running application and is not committed to this repository.

Do not upload sensitive production customer information to an environment unless that deployment has been appropriately secured and approved for the data.

## ⚠️ Limitations

- The model was trained on a single churn dataset.
- Risk bands are application-level categories, not separately trained classes.
- The default classification threshold is 50%.
- A probability is an estimate, not certainty.
- Model associations should not automatically be interpreted as causal relationships.
- Production use requires additional validation and monitoring.

For production deployments, consider authentication, authorization, rate limiting, model versioning, drift monitoring, audit logging, privacy controls and business-specific threshold calibration.

## 🛠️ Roadmap

- [ ] SHAP-based explanations
- [ ] Interactive feature sensitivity analysis
- [ ] Threshold tuning
- [ ] ROC-AUC / PR-AUC reporting
- [ ] Model comparison
- [ ] Prediction history
- [ ] Advanced dataset visualizations
- [ ] Authentication and user management
- [ ] Production deployment

---

<div align="center">

### ChurnIQ

**Turn customer data into churn-risk intelligence.**

Licensed under **AGPL-3.0**

</div>