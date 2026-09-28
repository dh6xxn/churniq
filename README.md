# ChurnIQ — Customer Churn Prediction using ANN

<p align="center">
  <strong>Predict customer churn from a customer profile using an Artificial Neural Network.</strong><br/>
  Full-stack academic / internship project with a FastAPI backend and responsive web frontend.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-ANN%2016--8--1-111827?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Test%20Accuracy-85.95%25-16a34a?style=for-the-badge" />
</p>

## About

**ChurnIQ** is the application version of the academic/internship project **Customer Churn Prediction using Artificial Neural Networks (ANN)**.

The project demonstrates how structured customer data can be transformed into a churn-risk prediction through preprocessing, feature encoding, scaling, neural-network inference, and a REST API.

The original presentation describes the project as an Artificial Intelligence with Python internship capstone completed through **SkillForge**, with the work focused on customer churn prediction using TensorFlow/Keras. The application in this repository turns that model concept into a usable end-to-end web application.

> **Purpose:** academic demonstration, learning, experimentation, and predictive-analytics prototyping. The output is a probability estimate, not a guarantee or a causal statement about a customer.

## What ChurnIQ does

1. Accepts a customer profile.
2. Validates the input.
3. Applies the same preprocessing pipeline used during model training.
4. Runs the profile through the ANN-style classifier.
5. Returns a churn probability from 0–100%.
6. Converts the probability into a binary prediction using a 50% threshold.
7. Displays a presentation-oriented Low / Medium / High risk band.

### Core workflow

```
Customer profile
      ↓
Input validation
      ↓
Categorical encoding + numerical scaling
      ↓
ANN / MLP: 16 → 8 → 1
      ↓
Churn probability
      ↓
Prediction + risk band
```

## Features used

The model uses these 10 customer attributes:

| Feature | Type | What it represents |
|---|---|---|
| CreditScore | Numeric | Customer credit score |
| Geography | Categorical | France, Germany, or Spain |
| Gender | Categorical | Male or Female |
| Age | Numeric | Customer age |
| Tenure | Numeric | Relationship duration |
| Balance | Numeric | Account balance |
| NumOfProducts | Numeric | Number of products held |
| HasCrCard | Binary | Credit-card ownership |
| IsActiveMember | Binary | Customer activity/engagement |
| EstimatedSalary | Numeric | Estimated salary |

The original dataset contains 10,000 records and the target variable `Exited` indicates whether the customer left.

## How feature changes affect risk

The model does **not** contain rules such as “+10 years of age = +X% churn.” It learns interactions among features.

Examples of what the features contribute:

- **Age:** the learned relationship can vary across age ranges.
- **Geography:** regional differences are learned from the encoded geography values.
- **Gender:** contributes as a categorical input alongside the other variables.
- **CreditScore:** provides a credit-profile signal.
- **Tenure:** captures relationship duration.
- **Balance:** adds account-balance information.
- **NumOfProducts:** represents product relationship depth and can have non-linear effects.
- **HasCrCard:** indicates credit-card ownership.
- **IsActiveMember:** captures customer engagement.
- **EstimatedSalary:** contributes an income-related signal.

### Important interpretation rule

A feature's model association is **not automatically a causal effect**.

For example, changing only Age in the UI and observing a different probability does not prove that age caused the difference. The neural network is estimating a joint prediction based on the complete profile.

For reliable feature-attribution work, methods such as SHAP or controlled sensitivity analysis should be added separately.

## Model

The application uses an ANN-shaped multilayer perceptron:

```
Input features
      │
      ▼
Dense / Hidden Layer — 16 neurons — ReLU
      │
      ▼
Dense / Hidden Layer — 8 neurons — ReLU
      │
      ▼
Output — 1 neuron — binary prediction
```

The shipped runnable model uses **scikit-learn MLPClassifier** so the web app does not require TensorFlow just to run.

The repository also contains `backend/train_keras.py`, which preserves the original TensorFlow/Keras implementation:

- Dense(16, ReLU)
- Dense(8, ReLU)
- Dense(1, Sigmoid)
- Adam optimizer
- Binary cross-entropy
- 50 epochs
- Batch size 32

## Preprocessing

The original implementation was corrected to avoid test-set leakage.

The correct order is:

```
Raw data
  ↓
Train / test split
  ↓
Fit preprocessing on training data only
  ↓
Transform training + test data
  ↓
Train ANN
  ↓
Evaluate on untouched test data
```

The runnable pipeline uses:

- One-hot encoding for `Geography`
- Binary encoding for `Gender`
- StandardScaler for numeric features
- 80/20 train/test split
- `random_state=42`
- Stratification of the target

## Evaluation results

The current trained model produced the following held-out test-set results:

| Metric | Result |
|---|---:|
| Accuracy | **85.95%** |
| Churn precision | **76.92%** |
| Churn recall | **44.23%** |
| Churn F1 | **56.16%** |
| Training records | 8,000 |
| Test records | 2,000 |
| Dataset records | 10,000 |

### Confusion matrix

```
                 Predicted
               Non-churn  Churn
Actual Non       1539       54
Actual Churn      227      180
```

The **44.23% churn recall** is important: the model does not identify every actual churn case at the default 50% threshold. Therefore accuracy should not be considered the only performance measure.

## Application

### Frontend

- Responsive customer profile form
- Churn probability display
- Circular probability visualization
- Risk band
- Prediction class
- Threshold display
- Model metrics
- Project explanation

### Backend

FastAPI provides:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Check API/model health |
| GET | `/api/model-info` | Return model metadata and metrics |
| POST | `/api/predict` | Predict churn for a customer profile |

Example request:

```json
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
```

## Project structure

```text
churniq/
├── backend/
│   ├── main.py
│   ├── train_model.py
│   ├── train_keras.py
│   └── model/
│       ├── churn_ann.joblib
│       └── metrics.json
├── data/
│   └── README.md
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── .gitignore
├── README.md
└── requirements.txt
```

## Run locally

### 1. Clone

```bash
git clone https://github.com/dh6xxn/churniq.git
cd churniq
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the application

The repository includes a trained model artifact, so the app can be started directly:

```bash
python -m uvicorn backend.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

### Retraining

Place `Churn_Modelling.csv` inside `data/`, then run:

```bash
python backend/train_model.py
```

This regenerates the model and metrics.

## Academic context

**Project:** Customer Churn Prediction using Artificial Neural Networks (ANN)

**Internship domain:** Artificial Intelligence with Python

**Internship organisation:** SkillForge

**Institution:** SRM Institute of Science and Technology, Ramapuram

**Internship period documented in the presentation:** May 1, 2025 – June 30, 2025

The accompanying project presentation describes the original workflow as data cleaning, categorical encoding, feature scaling, ANN model construction, training, and evaluation using accuracy, confusion matrix, precision, recall, and F1-score.

## Limitations

- The model is trained on one structured customer dataset.
- The displayed risk bands are UI categories, not trained classes.
- A 50% threshold is used for the binary prediction.
- Predictive association should not be interpreted as causation.
- The current application is an academic/demo system, not a production retention decision engine.
- Production deployment would require authentication, rate limiting, monitoring, model versioning, drift detection, privacy controls, and a validated business threshold.

## Future improvements

- SHAP-based feature explanations
- Interactive sensitivity analysis
- Threshold tuning for recall/precision trade-offs
- ROC-AUC and PR-AUC reporting
- Model comparison with Random Forest and XGBoost
- Dropout and hyperparameter tuning for the Keras model
- Prediction history
- Batch CSV prediction
- Authentication and production deployment

## License

This repository is intended as an academic/project demonstration. Add a project-specific license before redistributing it as an open-source package.
