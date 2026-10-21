"""Data Engine: preprocessing dei dati (gestione nulli, scaling e codifica)."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

PIMA_ZERO_AS_NAN = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


class DataEngine:
    """
    Preprocessing per il VAE:
      1. gestione dei valori nulli (zeri impossibili -> NaN -> mediana / moda)
      2. MinMaxScaler sulle variabili numeriche
      3. OneHotEncoder sulle variabili categoriche
    """

    def __init__(self, categorical_cols=None, zero_as_nan_cols=None, decimals=None):
        self.categorical_cols = categorical_cols or []
        self.zero_as_nan_cols = zero_as_nan_cols or []
        self.decimals = decimals or {}
        self.numeric_cols = []
        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.medians = {}
        self.modes = {}
        self.cat_group_sizes = []
        self.is_fitted = False

    def _handle_nulls(self, df):
        df = df.copy()
        for col in self.zero_as_nan_cols:
            df[col] = df[col].replace(0, np.nan)
        for col, median in self.medians.items():
            df[col] = df[col].fillna(median)
        for col, mode in self.modes.items():
            df[col] = df[col].fillna(mode)
        return df

    def fit(self, df):
        self.numeric_cols = [c for c in df.columns if c not in self.categorical_cols]

        tmp = df.copy()
        for col in self.zero_as_nan_cols:
            tmp[col] = tmp[col].replace(0, np.nan)
        self.medians = tmp[self.numeric_cols].median().to_dict()
        self.modes = {c: tmp[c].mode().iloc[0] for c in self.categorical_cols}

        clean = self._handle_nulls(df)
        self.scaler.fit(clean[self.numeric_cols])
        if self.categorical_cols:
            self.encoder.fit(clean[self.categorical_cols])
            self.cat_group_sizes = [len(c) for c in self.encoder.categories_]

        self.is_fitted = True
        return self

    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError("Chiama fit() prima di transform().")
        clean = self._handle_nulls(df)
        num = self.scaler.transform(clean[self.numeric_cols])
        if self.categorical_cols:
            cat = self.encoder.transform(clean[self.categorical_cols])
            return np.hstack([num, cat]).astype(np.float32)
        return num.astype(np.float32)

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def inverse_transform(self, X):
        n_num = len(self.numeric_cols)
        num = np.clip(X[:, :n_num], 0, 1)
        num = self.scaler.inverse_transform(num)
        out = pd.DataFrame(num, columns=self.numeric_cols)

        for col, d in self.decimals.items():
            if col in out.columns:
                out[col] = out[col].round(d)
                if d == 0:
                    out[col] = out[col].astype(int)

        if self.categorical_cols:
            cat_values = self.encoder.inverse_transform(X[:, n_num:])
            for i, col in enumerate(self.categorical_cols):
                out[col] = cat_values[:, i]
        return out

    @property
    def n_features(self):
        return len(self.numeric_cols) + sum(self.cat_group_sizes)


def infer_decimals(df, numeric_cols, max_decimals=4):
    """Per ogni colonna numerica trova il numero minimo di decimali che la rappresenta."""
    decimals = {}
    for col in numeric_cols:
        s = df[col].dropna().astype(float)
        for d in range(max_decimals + 1):
            if np.allclose(s, s.round(d)):
                decimals[col] = d
                break
        else:
            decimals[col] = max_decimals
    return decimals


def suggest_config(df):
    """
    Suggerimenti automatici per un dataset caricato dall'utente:
    - categoriche: colonne testuali/booleane o intere binarie (es. 0/1)
    - zero = nullo: solo se il dataset è il Pima Indians Diabetes
    """
    categorical = []
    for col in df.columns:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            categorical.append(col)
        elif pd.api.types.is_integer_dtype(s) and s.nunique() <= 2:
            categorical.append(col)
    zero_as_nan = PIMA_ZERO_AS_NAN if set(PIMA_ZERO_AS_NAN).issubset(df.columns) else []
    return categorical, zero_as_nan
