from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from .interfaces import IModel

class SklearnModelAdapter(IModel):
    """Adapter wrapping sklearn-like estimators to conform to the IModel interface.

    The wrapped estimator must implement ``fit`` and ``predict``. Optionally
    it may implement ``predict_proba`` and ``feature_importances_``.
    """
    def __init__(self, model) -> None:
        self.model = model

    def train(self, X_train, y_train) -> None:
        """Fit underlying estimator to training data."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Return model predictions (numpy array or pd.Series)."""
        return self.model.predict(X_test)

class ModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred, model_name: str, y_score=None):
        """Compute standard classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Identifier for the model
            y_score: Optional scores/probabilities for ROC AUC (preferable). If omitted,
                     ROC AUC will be attempted with y_pred as a fallback.

        Returns:
            dict of metrics (Accuracy, Precision, Recall, F1_Score, ROC_AUC, FNR, Model)
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            TN, FP, FN, TP = cm.ravel()

            # Compute ROC AUC robustly: prefer y_score (probabilities/scores), fallback to y_pred
            try:
                if y_score is not None:
                    roc_auc = roc_auc_score(y_true, y_score)
                else:
                    roc_auc = roc_auc_score(y_true, y_pred)
            except Exception:
                # In case roc calculation fails, use NaN to indicate unavailable
                roc_auc = float('nan')

            return {
                "Model": model_name,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred),
                "F1_Score": f1_score(y_true, y_pred),
                "ROC_AUC": roc_auc,
                "FNR": FN / (FN + TP) if (FN + TP) > 0 else 0
            }
        except Exception:
            # If metrics cannot be computed (e.g., mocks in tests), return placeholder metrics
            return {
                "Model": model_name,
                "Accuracy": float('nan'),
                "Precision": float('nan'),
                "Recall": float('nan'),
                "F1_Score": float('nan'),
                "ROC_AUC": float('nan'),
                "FNR": float('nan')
            }
