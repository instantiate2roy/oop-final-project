from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..exceptions import ModelError


@dataclass(frozen=True)
class ModelEvaluation:
    """How well a model ranks risk on data it wasn't trained on."""
    auc: float                 # 0.5 is guessing, 1.0 is perfect ranking
    rows: int
    defaults: int
    default_rate: float
    bands: pd.DataFrame        # default rate in each score band, lowest risk first


class RiskModel(ABC):
    """
    A model that gives each loan a probability of default (PD).

    Subclasses decide how the model is built and fitted; RiskModel provides evaluation,
    score bands, saving and loading, which work the same for any model.
    """

    name: str = "risk model"

    def __init__(self, features: list[str]):
        self.features = features
        self.trained = False

    @abstractmethod
    def fit(self, frame: pd.DataFrame, label: str = "defaulted") -> "RiskModel":
        """Train on a frame holding the feature columns and the label."""

    @abstractmethod
    def predict_pd(self, frame: pd.DataFrame) -> pd.Series:
        """Probability of default for each row, between 0 and 1."""

    def _check_trained(self) -> None:
        if not self.trained:
            raise ModelError(f"The {self.name} hasn't been trained yet; call fit() first")

    def _check_columns(self, frame: pd.DataFrame) -> None:
        missing = [column for column in self.features if column not in frame.columns]
        if missing:
            raise ModelError(f"Missing feature columns: {', '.join(missing)}")

    def evaluate(self, frame: pd.DataFrame, label: str = "defaulted", bands: int = 5) -> ModelEvaluation:
        """AUC, plus the actual default rate in each score band (A = lowest predicted risk)."""
        pd_values = self.predict_pd(frame)
        actual = frame[label]
        if actual.nunique() < 2:
            raise ModelError("Evaluation data needs both defaults and non-defaults")
        table = pd.DataFrame({"pd": pd_values.values, "defaulted": actual.values})
        table["band"] = pd.qcut(table["pd"].rank(method="first"), bands,
                                labels=[chr(65 + i) for i in range(bands)])
        summary = table.groupby("band", observed=True).agg(
            loans=("defaulted", "size"), defaults=("defaulted", "sum"),
            average_pd=("pd", "mean"), lowest_pd=("pd", "min"), highest_pd=("pd", "max"))
        summary["default_rate"] = summary["defaults"] / summary["loans"]
        return ModelEvaluation(auc=roc_auc_score(actual, pd_values), rows=len(table),
                               defaults=int(actual.sum()), default_rate=float(actual.mean()),
                               bands=summary.reset_index())

    def save(self, path: str | Path) -> Path:
        """Save the trained model to a file, creating its folder if needed."""
        self._check_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "RiskModel":
        """Load a model saved with save(). Only load files you created yourself."""
        path = Path(path)
        if not path.is_file():
            raise ModelError(f"Model file not found: {path.resolve()}")
        model = joblib.load(path)
        if not isinstance(model, RiskModel):
            raise ModelError(f"{path.name} doesn't contain a risk model")
        return model
