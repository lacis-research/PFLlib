"""Policy ported from mininetfed_iwcmc/mininetfed/server/straggler.py."""
import numpy as np


class OnlineStragglerPredictor:
    def __init__(
        self,
        feature_order=None,
        ridge=1.0,
        temperature=0.15,
        time_threshold=0.6,
    ):
        self.feature_order = feature_order or [
            "f_i",
            "dataset_size",
            "c_r_i",
            "b_mean",
            "energy_consumption",
            "round_time",
        ]
        self.dim = len(self.feature_order)
        self.A = np.eye(self.dim, dtype=np.float64) * float(ridge)
        self.b = np.zeros(self.dim, dtype=np.float64)
        self.temperature = max(float(temperature), 1e-6)
        self.time_threshold = float(time_threshold)
        self.update_count = 0

    def _clip(self, value):
        return max(0.0, min(1.0, float(value)))

    def vectorize_context(self, context):
        context = context or {}
        return np.asarray(
            [self._clip(context.get(feature, 0.0)) for feature in self.feature_order],
            dtype=np.float64,
        )

    def predict_round_time(self, context):
        x = self.vectorize_context(context)
        theta = np.linalg.solve(self.A, self.b)
        return self._clip(float(x.T @ theta))

    def predict_straggler_probability(self, context, threshold=None):
        predicted_round_time = self.predict_round_time(context)
        reference = self.time_threshold if threshold is None else float(threshold)
        z_value = (predicted_round_time - reference) / self.temperature
        probability = 1.0 / (1.0 + np.exp(-z_value))
        return {
            "predicted_round_time": float(predicted_round_time),
            "straggler_probability": self._clip(probability),
            "threshold": float(reference),
            "temperature": float(self.temperature),
            "num_updates": int(self.update_count),
        }

    def update(self, context, observed_round_time):
        x = self.vectorize_context(context)
        y = self._clip(observed_round_time)
        self.A += np.outer(x, x)
        self.b += y * x
        self.update_count += 1
        return {
            "observed_round_time": float(y),
            "num_updates": int(self.update_count),
        }
