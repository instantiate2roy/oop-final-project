import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..exceptions import ModelError
from .risk_model import RiskModel


class LogisticPdModel(RiskModel):
    """
    Probability of default from a logistic regression.

    Numbers are scaled so their coefficients can be compared; categories such as the loan
    product are turned into yes/no columns. Each coefficient says how a feature moves the risk.
    """

    name = "logistic PD model"

    def __init__(self, numeric: list[str], categorical: list[str], regularisation: float = 1.0):
        super().__init__(numeric + categorical)
        self.numeric = numeric
        self.categorical = categorical
        self.pipeline = Pipeline([
            ("prepare", ColumnTransformer([
                ("numbers", StandardScaler(), numeric),
                ("categories", OneHotEncoder(handle_unknown="ignore"), categorical),
            ])),
            ("model", LogisticRegression(C=regularisation, max_iter=2000)),
        ])

    def fit(self, frame: pd.DataFrame, label: str = "defaulted") -> "LogisticPdModel":
        self._check_columns(frame)
        if frame[label].nunique() < 2:
            raise ModelError("Training data needs both defaults and non-defaults")
        self.pipeline.fit(frame[self.features], frame[label])
        self.trained = True
        return self

    def predict_pd(self, frame: pd.DataFrame) -> pd.Series:
        self._check_trained()
        self._check_columns(frame)
        return pd.Series(self.pipeline.predict_proba(frame[self.features])[:, 1], index=frame.index, name="pd")

    def coefficients(self) -> pd.DataFrame:
        """Each feature's effect: positive raises the predicted risk, negative lowers it."""
        self._check_trained()
        names = self.pipeline.named_steps["prepare"].get_feature_names_out()
        names = [n.split("__", 1)[1] for n in names]
        weights = self.pipeline.named_steps["model"].coef_[0]
        table = pd.DataFrame({"feature": names, "coefficient": weights})
        table["effect"] = np.where(table["coefficient"] > 0, "raises risk", "lowers risk")
        return table.reindex(table["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)
