# ChurnIQ

> Customer churn prediction and bulk risk analysis powered by an Artificial Neural Network.

ChurnIQ turns a trained ANN/MLP churn model into a usable web application. It supports both **single-customer prediction** and **CSV-based bulk analysis**, so you can upload a customer dataset and quickly identify the customers with the highest predicted churn probability.

## Features

- Individual customer churn prediction
- CSV dataset upload and bulk prediction
- Automatic CSV validation
- Churn probability for every customer
- Low / Medium / High risk bands
- Highest-risk customer ranking
- Downloadable prediction results
- Model performance dashboard
- FastAPI REST API
- Responsive frontend
- Correct train/test preprocessing without test-set leakage

## How it works

```
Customer data
    │
    ├── Single customer ──► validation ──► preprocessing ──► ANN ──► risk
    │
    └── CSV dataset ──────► validation ──► preprocessing ──► ANN ──► ranked results
```

The model uses an MLP/ANN architecture equivalent to:

```
Input
  ↓
16 neurons — ReLU
  ↓
8 neurons — ReLU
  ↓
1 neuron — binary classification
```

The runnable application uses scikit-learn's `MLPClassifier`. `backend/train_keras.py` contains the TensorFlow/Keras implementation of the same 16 → 8 → 1 architecture.

## Model features

| Feature | Description |
|---|---|
| CreditScore | Customer credit score |
| Geography | France, Germany or Spain |
| Gender | Male or Female |
| Age | Customer age |
| Tenure | Years with the institution |
| Balance | Account balance |
| NumOfProducts | Number of products held |
| HasCrCard | Credit-card ownership |
| IsActiveMember | Customer activity status |
| EstimatedSalary | Estimated salary |

Identifier columns such as `CustomerId`, `Surname`, and `RowNumber` may be retained in an uploaded CSV for identification, but they are not model features.

### How feature changes affect risk

The ANN learns relationships between the features rather than applying simple fixed rules. A change to Age, Balance, Geography, activity, product count, or another input can therefore change the predicted probability depending on the complete customer profile.

A prediction change is **not a causal conclusion**. For example, if changing one field changes the probability, that does not prove that the field caused churn. Feature-attribution methods such as SHAP can be added when causal/interpretability analysis is required.

## Bulk CSV analysis

Open **Dataset Analysis** and upload a CSV containing these required fields:

```text
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
```

Optional identifier fields:

```text
CustomerId
Surname
RowNumber
```

ChurnIQ will:

1. Read the CSV.
2. Validate required columns and values.
3. Report invalid rows instead of silently accepting bad data.
4. Generate a churn probability for every valid row.
5. Assign a Low, Medium or High risk band.
6. Sort customers by predicted churn probability.
7. Show summary counts and high-risk customers.
8. Let you download the complete prediction CSV.

The output adds:

```text
ChurnProbability
ChurnPercentage
Prediction
Risk
```

**The uploaded customer data is processed by the running application and is not included in this repository.**

## Evaluation

Current held-out test-set results:

| Metric | Result |
|---|---:|
| Accuracy | 85.95% |
| Churn precision | 76.92% |
| Churn recall | 44.23% |
| Churn F1 | 56.16% |
| Training records | 8,000 |
| Test records | 2,000 |

Confusion matrix:

```
                 Predicted
               Non-churn  Churn
Actual Non       1539       54
Actual Churn      227      180
```

The default classification threshold is **50%**. The probability is more informative than the binary class when prioritising customers for review.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | API/model health |
| GET | `/api/model-info` | Model metrics |
| POST | `/api/predict` | Single-customer prediction |
| POST | `/api/predict-csv` | Bulk CSV prediction |

## Run locally

### 1. Clone

```bash
git clone https://github.com/dh6xxn/churniq.git
cd churniq
```

### 2. Create an environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start

```bash
python -m uvicorn backend.main:app --reload
```

Open **http://127.0.0.1:8000**

### Retrain

Put `Churn_Modelling.csv` in `data/` and run:

```bash
python backend/train_model.py
```

The raw dataset is intentionally not tracked by Git.

## Project structure

```text
churniq/
├── backend/
│   ├── main.py
│   ├── train_model.py
│   ├── train_keras.py
│   └── model/
│       └── metrics.json
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── data/
│   └── README.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Limitations

- The model was trained on a single churn dataset.
- Risk bands are application-level categories, not separate trained classes.
- A 50% threshold is used for the binary prediction.
- Model probability is an estimate, not certainty.
- The application is not a substitute for validated production retention systems.
- Production use should add authentication, access control, monitoring, model versioning, drift checks, privacy controls and threshold validation.

## License

ChurnIQ is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [LICENSE](LICENSE).
