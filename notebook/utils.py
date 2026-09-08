from typing import ClassVar

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RANDOM_STATE = 42

class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Reproduces every cleaning + feature-engineering step from your notebook,
    starting from the raw column schema."""

    SERVICE_COLS: ClassVar[list[str]] = [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    BINARY_COLS: ClassVar[list[str]] = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    YN_MAP: ClassVar[dict[str, int]] = {'Yes': 1, 'No': 0}
    GENDER_MAP: ClassVar[dict[str, int]] = {'Male': 1, 'Female': 0}
    def fit(self, X, y=None):
        return self  # nothing here needs to be learned from data

    def transform(self, X):
        df = X.copy()

        # Cleaning
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        str_cols = df.select_dtypes(include='str').columns
        df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

        # Binary mappings
        for col in self.BINARY_COLS:
            df[col] = df[col].map(self.YN_MAP)
        df['gender'] = df['gender'].map(self.GENDER_MAP)

        # Engineered features
        df['TenureGroup'] = df['tenure'].apply(self._tenure_group)
        df['TotalServices'] = df.apply(self._count_services, axis=1)
        df['AvgMonthlyCharge'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
        df['ChargeDeviation'] = df['MonthlyCharges'] - df['AvgMonthlyCharge']

        return df.drop(columns=['customerID'], errors='ignore')

    @staticmethod
    def _tenure_group(t):
        if t <= 12: return '0-1yr'
        elif t <= 24: return '1-2yr'
        elif t <= 48: return '2-4yr'
        return '4yr+'

    def _count_services(self, row):
        return sum(
            row[col] not in ['No', 'No internet service', 'No phone service']
            for col in self.SERVICE_COLS
        )

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

multi_cat_cols = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod', 'TenureGroup'
]

preprocessor = ColumnTransformer(
    transformers=[('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), multi_cat_cols)],
    remainder='passthrough'   # everything else (numeric + already-binary columns) passes through unchanged
)

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

best_params = {'class_weight': None, 'criterion': 'gini', 'max_depth': 7, 'min_samples_leaf': 50} # as per gridcv

pipeline = Pipeline(steps=[
    ('feature_engineering', TelcoFeatureEngineer()),
    ('preprocessing', preprocessor),
    ('classifier', DecisionTreeClassifier(
        **best_params, # reuse best config
        random_state=RANDOM_STATE
    ))
])