# Telecom Customer Churn Prediction

An end-to-end machine-learning project for identifying telecom customers who are
at risk of churn. The project includes exploratory analysis, feature
engineering, a reusable scikit-learn pipeline, a saved model artifact, and a
FastAPI service for predictions on new customers.

## Project Workflow

The notebooks implement the workflow:

1. Load and inspect the IBM Telco Customer Churn dataset.
2. Clean numeric and categorical values, including blank `TotalCharges` values.
3. Explore churn patterns with visualizations.
4. Engineer tenure groups, service counts, and charge deviation features.
5. Train and compare Decision Tree classifiers using a stratified 70:30 split
	 and `random_state=42`.
6. Save the preprocessing and classifier together as a reusable pipeline.
7. Serve predictions through the FastAPI application in `main.py`.

The target is `Churn` (`Yes` or `No`). The dataset is imbalanced, with a
churn rate of approximately 26.5%, so recall, precision, F1 score, and ROC-AUC
are considered alongside accuracy. For the retention use case, recall is
prioritized because missing a likely churner can be more costly than contacting
a customer who would have stayed.

## Repository Structure

```text
.
├── data/
│   ├── TelcoCustomerChurn.csv
│   ├── TelcoCustomerChurn - Data Dictionary.csv
│   └── assignment.md
├── docs/
│   └── project-submission.md       # Detailed analysis and evaluation notes
├── model/
│   └── telco_churn_pipeline_latest.joblib
├── notebook/
│   ├── churn_analysis.ipynb         # Detailed EDA and model development
│   └── pipeline_based_analysis.ipynb
├── utils/
│   └── utils.py                     # Feature engineering and sklearn pipeline
├── main.py                          # FastAPI application
└── pyproject.toml                   # Python 3.12+ dependencies
```

## Setup

The project uses Python 3.12 or newer and `uv` for environment and dependency
management.

```bash
uv sync
source .venv/bin/activate
```

The model artifact is already included in `model/`, so retraining is not
required to start the API.

## Run the API

From the project root:

```bash
uv run uvicorn main:app --reload
```

The service is available at `http://localhost:8000`. Interactive OpenAPI
documentation is available at `http://localhost:8000/docs`.

### Health check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### Predict churn

`POST /predict` accepts the customer fields below. `customerID` is required by
the request schema but is discarded before prediction because it is an
identifier, not a predictive feature.

```bash
curl -X POST http://localhost:8000/predict \
	-H "Content-Type: application/json" \
	-d '{
		"customerID": "9999-XYZ",
		"gender": "Female",
		"SeniorCitizen": 0,
		"Partner": "Yes",
		"Dependents": "No",
		"tenure": 5,
		"PhoneService": "Yes",
		"MultipleLines": "No",
		"InternetService": "Fiber optic",
		"OnlineSecurity": "No",
		"OnlineBackup": "No",
		"DeviceProtection": "No",
		"TechSupport": "No",
		"StreamingTV": "No",
		"StreamingMovies": "No",
		"Contract": "Month-to-month",
		"PaperlessBilling": "Yes",
		"PaymentMethod": "Electronic check",
		"MonthlyCharges": 85.0,
		"TotalCharges": 425.0
	}'
```

The response has this shape:

```json
{
	"prediction": "Yes",
	"churn_probability": 0.82
}
```

Invalid or missing fields are rejected by the Pydantic request model with a
validation error.

## Run the Analysis

Open either notebook in VS Code with the Jupyter extension and select the
environment created by `uv sync` as the kernel. The notebooks read the dataset
from `data/TelcoCustomerChurn.csv`. The pipeline-based notebook demonstrates
how the same cleaning and feature-engineering steps are applied consistently
to unseen API input.

The reusable pipeline in `utils/utils.py` performs the following transformations:

- Converts `TotalCharges` to numeric and fills blank values.
- Trims whitespace and maps binary and gender values.
- Creates `TenureGroup`, `TotalServices`, `AvgMonthlyCharge`, and
	`ChargeDeviation`.
- One-hot encodes multi-category features and ignores unknown categories.
- Applies a Decision Tree classifier to the transformed data.

## Model Results

The documented comparison found that the unconstrained tree overfit the
training data. The recall-oriented constrained tree was selected for the
retention use case:

| Metric | Test result |
| --- | ---: |
| Accuracy | 69.5% |
| Precision (Churn) | 46.0% |
| Recall (Churn) | 85.6% |
| F1 score (Churn) | 59.8% |
| ROC-AUC | 0.831 |

These results are evaluation notes from the analysis notebooks, not a promise
that every future retraining run will produce identical values. The checked-in
joblib file is the artifact loaded by the API.

## Additional Documentation

- [Detailed submission notes](docs/project-submission.md)
- [Assignment requirements](data/assignment.md)
