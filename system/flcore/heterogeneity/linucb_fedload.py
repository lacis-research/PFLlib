"""Policy ported from mininetfed_iwcmc/mininetfed/server/bandits/linucb_fedload.py."""
import numpy as np


class LinUCBBandit:
    def __init__(
        self,
        pruning_rates,
        feature_order=None,
        alpha=0.25,
        beta=0.6,
        ridge=1.0,
        random_seed=42,
        min_alpha_ratio=0.0,
        alpha_strategy="proportional"
    ):
        self.pruning_rates = [float(rate) for rate in pruning_rates]
        self.feature_order = feature_order or [
            "e_cmp",
            "e_up",
            "e_down",
            "f_i",
            "dataset_size",
            "c_r_i",
            "b_mean"
        ]
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.min_alpha_ratio = float(min_alpha_ratio)
        self.alpha_strategy = str(alpha_strategy)
        self.dim = len(self.feature_order)
        self.rng = np.random.default_rng(random_seed)
        self.A = {
            arm: np.eye(self.dim, dtype=np.float64) * ridge for arm in self.pruning_rates
        }
        self.b = {
            arm: np.zeros(self.dim, dtype=np.float64) for arm in self.pruning_rates
        }
        self.arm_counts = {arm: 0 for arm in self.pruning_rates}

    def vectorize_context(self, context):
        return np.asarray(
            [max(0.0, min(1.0, float(context.get(feature, 0.0)))) for feature in self.feature_order],
            dtype=np.float64
        )

    def compute_alpha_r_i(self, pruning_reference=None, candidate_arm=None):
        if isinstance(pruning_reference, (list, tuple, np.ndarray)):
            pruning_values = [float(value) for value in pruning_reference]
            avg_pruning_before_choice = float(np.mean(pruning_values)) if pruning_values else 1.0
            if candidate_arm is not None:
                pruning_values.append(float(candidate_arm))
            avg_pruning = float(np.mean(pruning_values)) if pruning_values else 1.0
        else:
            avg_pruning = float(pruning_reference) if pruning_reference is not None else 1.0
            avg_pruning_before_choice = avg_pruning
        if self.alpha_strategy == "binary":
            exploration_ratio = 1.0 if avg_pruning >= self.beta else self.min_alpha_ratio
        else:
            if self.beta <= 0:
                scaled_ratio = 1.0
            else:
                scaled_ratio = max(0.0, min(1.0, avg_pruning / self.beta))
            exploration_ratio = max(self.min_alpha_ratio, scaled_ratio)
        alpha_r_i = self.alpha * exploration_ratio
        return {
            "alpha_r_i": float(alpha_r_i),
            "avg_pruning_for_alpha": float(avg_pruning),
            "avg_pruning_before_choice": float(avg_pruning_before_choice),
            "beta": float(self.beta),
            "exploration_ratio": float(exploration_ratio),
            "alpha_strategy": self.alpha_strategy
        }

    def choose_pruning_rate(self, context, pruning_reference=None, pending_arm_counts=None):
        x = self.vectorize_context(context)

        best_arms = []
        best_score = float("-inf")
        arm_alpha_info = {}
        for arm in self.pruning_rates:
            alpha_info = self.compute_alpha_r_i(pruning_reference, candidate_arm=arm)
            current_alpha = alpha_info["alpha_r_i"]
            arm_alpha_info[arm] = alpha_info
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            exploit = float(x.T @ theta)
            explore = current_alpha * float(np.sqrt(x.T @ A_inv @ x))
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_arms = [arm]
            elif np.isclose(score, best_score):
                best_arms.append(arm)
        best_arm = float(self.rng.choice(best_arms))
        alpha_info = dict(arm_alpha_info[best_arm])
        alpha_info["selected_pruning_rate"] = float(best_arm)
        alpha_info["selection_mode"] = "linucb"
        return float(best_arm), alpha_info

    def update(self, chosen_rate, reward, context):
        arm = float(chosen_rate)
        x = self.vectorize_context(context)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += float(reward) * x
        self.arm_counts[arm] += 1
