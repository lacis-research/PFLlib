"""Versioned, train-only semantic summaries; no sample embeddings leave clients."""
from dataclasses import dataclass, field
import json
import math
import numpy as np
import torch
from .linucb_fedload import LinUCBBandit

SCHEMA = 'fedload-semantic-v2'


@dataclass
class Bank:
    round: int = 0
    dimension: int = 0
    prototypes: dict = field(default_factory=dict)
    contributors: dict = field(default_factory=dict)
    version: str = SCHEMA


@dataclass
class Payload:
    client_id: int
    round: int
    retention: float
    dimension: int
    counts: dict
    prototypes: dict
    indicators: dict
    valid: dict
    coverage: float
    reference_round: int
    version: str = SCHEMA

    def metadata(self):
        return {k: v for k, v in vars(self).items() if k != 'prototypes'}


def wire_bytes(value):
    """Canonical UTF-8 JSON metadata + sorted int64 class IDs + float32 vectors."""
    metadata = {k: v for k, v in vars(value).items() if k != 'prototypes'}
    header = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return 8 + len(header) + sum(8 + p.numel() * 4 for p in value.prototypes.values())


@torch.no_grad()
def extract(model, loader, classes, client_id, round_index, retention, bank, device):
    mode = model.training
    model.eval()
    sums, squares, counts = {}, {}, {}
    uncertainty = 0.0
    try:
        for x, y in loader:
            logits, features = model.forward_with_features(x.to(device))
            probabilities = logits.softmax(1)
            uncertainty += float((-(probabilities * probabilities.clamp_min(1e-30).log()).sum(1)).sum())
            features = features.detach().cpu().double()
            for label in y.unique().tolist():
                values = features[y.cpu() == label]
                sums[label] = sums.get(label, torch.zeros(features.shape[1], dtype=torch.float64)) + values.sum(0)
                squares[label] = squares.get(label, 0.) + float(values.square().sum())
                counts[label] = counts.get(label, 0) + len(values)
    finally:
        model.train(mode)
    total = sum(counts.values())
    prototypes = {k: (v / counts[k]).float() for k, v in sums.items()}
    dimension = model.fc.in_features
    entropy = -sum(n / total * math.log(n / total) for n in counts.values()) if total else 0.
    log_classes = math.log(classes) if classes > 1 else 1.
    dispersion = sum(max(0., squares[k] - float(sums[k].square().sum()) / counts[k]) for k in counts)
    magnitude = sum(squares.values())
    quality = 1 - dispersion / magnitude if magnitude > 1e-12 else 0.
    divergence, shared = 0., 0
    if bank.version != SCHEMA or (bank.prototypes and bank.dimension != dimension):
        raise ValueError('Incompatible semantic bank feature schema/dimension.')
    for k, p in prototypes.items():
        ref = bank.prototypes.get(k)
        if ref is not None and p.norm() > 1e-12 and ref.norm() > 1e-12:
            divergence += counts[k] * float((1 - torch.nn.functional.cosine_similarity(p, ref, dim=0)).clamp(0, 2) / 2)
            shared += counts[k]
    return Payload(client_id, round_index, retention, dimension, counts, prototypes,
                   dict(H=entropy/log_classes, U=uncertainty/max(total, 1)/log_classes,
                        D=divergence/max(shared, 1), Q=max(0., min(1., quality))),
                   dict(H=total > 0, U=total > 0, D=shared > 0, Q=magnitude > 1e-12),
                   shared/max(total, 1), bank.round)


def build_bank(payloads, previous, round_index):
    if not payloads:
        return previous
    sums, counts, contributors = {}, {}, {}
    dimension = payloads[0].dimension
    for p in payloads:
        if p.round != round_index or p.version != SCHEMA or p.dimension != dimension:
            raise ValueError('Incompatible semantic payload round/schema/dimension.')
        for k, proto in p.prototypes.items():
            sums[k] = sums.get(k, torch.zeros_like(proto)) + proto.detach().cpu() * p.counts[k]
            counts[k] = counts.get(k, 0) + p.counts[k]
            contributors.setdefault(k, []).append(p.client_id)
    return Bank(round_index, dimension, {k: (v/counts[k]).clone() for k, v in sums.items()}, contributors)


def history_context(payload, round_index, max_age, features):
    age = round_index - payload.round if payload else None
    available = payload is not None and 0 < age <= max_age and any(payload.valid[k] for k in features)
    context = dict(semantic_available=float(available), semantic_age=min(age/max(max_age, 1), 1.) if age is not None else 1.,
                   semantic_retention=payload.retention if payload else 0.,
                   semantic_coverage=payload.coverage if available and 'D' in features else 0.,
                   semantic_source_round=payload.round if payload else -1,
                   semantic_reference_round=payload.reference_round if payload else -1)
    for key in ('H', 'U', 'D', 'Q'):
        valid = available and key in features and payload.valid[key]
        if key == 'D' and payload is not None:
            valid = valid and 0 <= round_index-payload.reference_round <= max_age
        context['semantic_' + key] = payload.indicators[key] if valid else 0.
        context['semantic_valid_' + key] = float(valid)
    context['semantic_available'] = float(any(context['semantic_valid_' + k] for k in ('H', 'U', 'D', 'Q')))
    if not context['semantic_valid_D']:
        context['semantic_coverage'] = 0.
    return context


class SemanticLinUCB(LinUCBBandit):
    schema = SCHEMA

    def enable(self, policy='linucb', semantic_weight=0.6, cost_weight=0.1):
        self.policy = policy
        self.semantic_weight, self.cost_weight = semantic_weight, cost_weight
        self.base_dim = self.dim
        self.feature_order = self.feature_order + ['semantic_' + k for k in ('H', 'U', 'D', 'Q')] + ['semantic_valid_' + k for k in ('H', 'U', 'D', 'Q')] + ['semantic_age', 'semantic_retention', 'semantic_coverage', 'semantic_available']
        self.dim = len(self.feature_order)
        self.A = {a: np.eye(self.dim) for a in self.pruning_rates}
        self.b = {a: np.zeros(self.dim) for a in self.pruning_rates}

    def choose_semantic(self, context, reference, costs, fallback):
        feasible = [a for a in self.pruning_rates if costs[a]['feasible']]
        # Use the unextended FedLoad policy for warmup and unavailable history.
        original, legacy = self.legacy_policy.choose_pruning_rate(context, reference) if hasattr(self, "legacy_policy") else super().choose_pruning_rate(context, reference)
        eligible = feasible
        info = dict(legacy, schema=self.schema, feasible_arms=feasible, eligible_arms=eligible, legacy_eligible_arms=list(self.pruning_rates),
                    candidate_costs=costs, fallback=fallback, scores={})
        if not eligible:
            return None, dict(info, selection_mode='no_feasible_arm')
        x = self.vectorize_context(context)
        target = float(np.mean(self.pruning_rates))
        for arm in eligible:
            inv = np.linalg.inv(self.A[arm])
            theta = inv @ self.b[arm]
            semantic = float(x[self.base_dim:] @ theta[self.base_dim:])
            exploit = float(x[:self.base_dim] @ theta[:self.base_dim])
            explore = self.compute_alpha_r_i(reference, arm)['alpha_r_i'] * float(np.sqrt(x @ inv @ x))

            if self.policy == 'heuristic' and not fallback:
                # Reliability gates demand; entropy alone is not a difficulty ordering.
                demand = context['semantic_U'] * context['semantic_Q'] + context['semantic_D'] * (1-context['semantic_H'])
                desired = np.clip(target + self.semantic_weight * (demand - 0.5), min(self.pruning_rates), max(self.pruning_rates))
                semantic = -self.semantic_weight * abs(arm - desired)
                # Resource terms and exploration remain those of FedLoad.
            if fallback or self.policy == 'legacy':
                semantic = 0.
                if hasattr(self, 'legacy_policy'):
                    legacy_x = self.legacy_policy.vectorize_context(context)
                    legacy_inv = np.linalg.inv(self.legacy_policy.A[arm])
                    exploit = float(legacy_x @ legacy_inv @ self.legacy_policy.b[arm])
                    explore = self.compute_alpha_r_i(reference, arm)['alpha_r_i'] * float(np.sqrt(legacy_x @ legacy_inv @ legacy_x))
            cost = self.cost_weight * costs[arm]['fraction'] if self.policy == 'heuristic' and not fallback else 0.
            info['scores'][arm] = dict(estimated_reward=exploit, semantic=semantic, exploration=explore,
                                       cost=cost, total=exploit+semantic+explore-cost)
        chosen = max(eligible, key=lambda a: (info['scores'][a]['total'], -a))
        if (fallback or self.policy == 'legacy') and original in eligible:
            chosen = original
        info.update(self.compute_alpha_r_i(reference, chosen))
        info.update(selected_pruning_rate=chosen, selection_mode=self.policy)
        return chosen, info

    def state_dict(self):
        return dict(schema=self.schema, feature_order=self.feature_order, A=self.A, b=self.b)

    def load_state_dict(self, state):
        if state.get('schema') != self.schema or state.get('feature_order') != self.feature_order or set(state['A']) != set(self.A):
            raise ValueError('Incompatible semantic LinUCB checkpoint schema/features/arms.')
        for a in self.A:
            if np.shape(state['A'][a]) != (self.dim, self.dim) or np.shape(state['b'][a]) != (self.dim,):
                raise ValueError('Incompatible semantic LinUCB checkpoint dimensions.')
        self.A = {a: v.copy() for a, v in state['A'].items()}
        self.b = {a: v.copy() for a, v in state['b'].items()}


@torch.no_grad()
def classification_summary(model, loader, classes, device):
    """Classes without test support are unavailable, not zero recall."""
    confusion = torch.zeros(classes, classes, dtype=torch.int64)
    mode = model.training
    model.eval()
    try:
        for x, y in loader:
            predictions = model(x.to(device)).argmax(1).cpu()
            confusion += torch.bincount(y.cpu()*classes+predictions, minlength=classes*classes).reshape(classes, classes)
    finally:
        model.train(mode)
    return confusion


def summarize_confusion(confusion):
    support = confusion.sum(1)
    predicted = confusion.sum(0)
    tp = confusion.diag()
    present = support > 0
    recalls = tp.double()/support.clamp_min(1)
    f1 = 2*tp.double()/(support+predicted).clamp_min(1)
    return dict(accuracy=float(tp.sum()/support.sum()) if support.sum() else None,
                balanced_accuracy=float(recalls[present].mean()) if present.any() else None,
                macro_f1=float(f1[present].mean()) if present.any() else None,
                support=support.tolist(), recall=[float(recalls[i]) if present[i] else None for i in range(len(support))])
