"""Policy ported from mininetfed_iwcmc/mininetfed/server/bandits/linucb_hera.py."""
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
        alpha_strategy="proportional",
        capability_guidance=None,
    ):
        self.pruning_rates = [float(rate) for rate in pruning_rates]
        self.feature_order = feature_order or [
            "e_cmp",
            "e_up",
            "e_down",
            "f_i",
            "dataset_size",
            "c_r_i",
            "b_mean",
            "predicted_round_time",
            "straggler_risk",
        ]
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.min_alpha_ratio = float(min_alpha_ratio)
        self.alpha_strategy = str(alpha_strategy)
        self.capability_guidance = dict(capability_guidance or {})
        self.guidance_enabled = bool(self.capability_guidance.get("enabled", True))
        self.guidance_strength = float(self.capability_guidance.get("strength", 0.35))
        self.straggler_penalty = float(self.capability_guidance.get("straggler_penalty", 0.25))
        self.compute_weight = float(self.capability_guidance.get("compute_weight", 0.4))
        self.bandwidth_weight = float(self.capability_guidance.get("bandwidth_weight", 0.25))
        self.energy_weight = float(self.capability_guidance.get("energy_weight", 0.15))
        self.straggler_weight = float(self.capability_guidance.get("straggler_weight", 0.2))
        self.static_bias = float(self.capability_guidance.get("static_bias", 0.7))
        self.dynamic_penalty = float(self.capability_guidance.get("dynamic_penalty", 0.35))
        self.high_capacity_floor = float(self.capability_guidance.get("high_capacity_floor", 0.6))
        self.dim = len(self.feature_order)
        self.rng = np.random.default_rng(random_seed)
        self.A = {
            arm: np.eye(self.dim, dtype=np.float64) * ridge for arm in self.pruning_rates
        }
        self.b = {
            arm: np.zeros(self.dim, dtype=np.float64) for arm in self.pruning_rates
        }

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

    def choose_pruning_rate(self, context, pruning_reference=None):
        x = self.vectorize_context(context)

        best_arms = []
        best_score = float("-inf")
        arm_alpha_info = {}
        target_retention = self._compute_target_retention(context)
        for arm in self.pruning_rates:
            alpha_info = self.compute_alpha_r_i(pruning_reference, candidate_arm=arm)
            current_alpha = alpha_info["alpha_r_i"]
            arm_alpha_info[arm] = alpha_info
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            exploit = float(x.T @ theta)
            explore = current_alpha * float(np.sqrt(x.T @ A_inv @ x))
            guidance_bonus = self._capability_guidance_bonus(float(arm), target_retention, context)
            score = exploit + explore + guidance_bonus
            if score > best_score:
                best_score = score
                best_arms = [arm]
            elif np.isclose(score, best_score):
                best_arms.append(arm)
        best_arm = float(self.rng.choice(best_arms))
        alpha_info = dict(arm_alpha_info[best_arm])
        alpha_info["selected_pruning_rate"] = float(best_arm)
        alpha_info["selection_mode"] = "linucb"
        alpha_info["target_retention"] = float(target_retention)
        return float(best_arm), alpha_info

    def update(self, chosen_rate, reward, context):
        arm = float(chosen_rate)
        x = self.vectorize_context(context)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += float(reward) * x

    def _compute_target_retention(self, context):
        if not self.guidance_enabled:
            return float(np.mean(self.pruning_rates))

        static_compute_capacity = float(context.get("static_cpu_capacity", context.get("cpu_capacity", context.get("f_i", 0.5))))
        static_compute_efficiency = 1.0 - float(context.get("static_compute_cost", context.get("c_r_i", 0.5)))
        static_bandwidth_capacity = float(context.get("static_bandwidth", context.get("bandwidth", context.get("b_mean", 0.5))))
        static_energy_efficiency = float(context.get("static_energy_efficiency", context.get("energy_efficiency", 0.5)))

        runtime_compute_capacity = float(context.get("cpu_capacity", context.get("f_i", static_compute_capacity)))
        runtime_compute_efficiency = 1.0 - float(context.get("c_r_i", 0.5))
        runtime_bandwidth_capacity = float(context.get("bandwidth", context.get("b_mean", static_bandwidth_capacity)))
        runtime_energy_efficiency = float(context.get("energy_efficiency", static_energy_efficiency))
        straggler_risk = float(context.get("straggler_risk", 0.0))

        static_score = (
            self.compute_weight * ((static_compute_capacity + static_compute_efficiency) / 2.0) +
            self.bandwidth_weight * static_bandwidth_capacity +
            self.energy_weight * static_energy_efficiency
        )
        runtime_score = (
            self.compute_weight * ((runtime_compute_capacity + runtime_compute_efficiency) / 2.0) +
            self.bandwidth_weight * runtime_bandwidth_capacity +
            self.energy_weight * runtime_energy_efficiency
        )

        blended_score = (
            self.static_bias * static_score +
            (1.0 - self.static_bias) * runtime_score
        )
        degradation = max(0.0, static_score - runtime_score)
        capability_score = blended_score - (self.dynamic_penalty * degradation) - (self.straggler_weight * straggler_risk)
        if static_score >= 0.7:
            capability_score = max(capability_score, self.high_capacity_floor)
        capability_score = max(0.0, min(1.0, capability_score))

        min_arm = float(min(self.pruning_rates))
        max_arm = float(max(self.pruning_rates))
        return min_arm + capability_score * (max_arm - min_arm)

    def _capability_guidance_bonus(self, arm, target_retention, context):
        if not self.guidance_enabled:
            return 0.0

        min_arm = float(min(self.pruning_rates))
        max_arm = float(max(self.pruning_rates))
        arm_span = max(max_arm - min_arm, 1e-6)
        normalized_distance = abs(float(arm) - float(target_retention)) / arm_span
        bonus = self.guidance_strength * (1.0 - normalized_distance)

        straggler_risk = float(context.get("straggler_risk", 0.0))
        if arm > target_retention:
            oversize_ratio = (float(arm) - float(target_retention)) / arm_span
            bonus -= self.straggler_penalty * straggler_risk * oversize_ratio

        return float(bonus)
