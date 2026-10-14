import math
import random


class StratifiedRandom:
    def __init__(self):
        self.trainers_per_round = 5
        self.selection_fraction = None
        self.random_seed = 42
        self.client_contexts = {}
        self.initial_contexts = {}
        self._rng = random.Random(self.random_seed)

    def set_contexts(self, client_contexts=None, initial_contexts=None):
        self.client_contexts = dict(client_contexts or {})
        self.initial_contexts = dict(initial_contexts or {})

    def _get_target_size(self, trainer_list):
        trainers_per_round = self.trainers_per_round
        if self.selection_fraction is not None:
            trainers_per_round = max(1, math.floor(len(trainer_list) * float(self.selection_fraction)))
        return min(len(trainer_list), int(trainers_per_round))

    def _static_capacity(self, trainer_id):
        context = dict(self.initial_contexts.get(trainer_id, {}))
        if not context:
            context = dict(self.client_contexts.get(trainer_id, {}))
        return float(context.get("cpu_capacity", context.get("f_i", 0.5)))

    def _build_strata(self, trainer_list):
        ordered = sorted(
            [(trainer_id, self._static_capacity(trainer_id)) for trainer_id in trainer_list],
            key=lambda item: item[1],
        )
        if not ordered:
            return []

        if len(ordered) < 3:
            return [[trainer_id for trainer_id, _ in ordered]]

        base_size = len(ordered) // 3
        remainder = len(ordered) % 3
        sizes = [base_size] * 3
        for idx in range(remainder):
            sizes[idx] += 1

        strata = []
        cursor = 0
        for size in sizes:
            next_cursor = cursor + size
            strata.append([trainer_id for trainer_id, _ in ordered[cursor:next_cursor]])
            cursor = next_cursor
        return [group for group in strata if group]

    def select_trainers_for_round(self, trainer_list, metrics):
        target_size = self._get_target_size(trainer_list)
        if len(trainer_list) <= target_size:
            return list(trainer_list)

        strata = self._build_strata(trainer_list)
        if len(strata) <= 1:
            return self._rng.sample(list(trainer_list), target_size)

        selected = []
        available_groups = [list(group) for group in strata]

        while len(selected) < target_size:
            progress = False
            for group in available_groups:
                remaining_slots = target_size - len(selected)
                if remaining_slots <= 0:
                    break
                if not group:
                    continue
                if len(selected) >= target_size:
                    break
                choice = self._rng.choice(group)
                group.remove(choice)
                selected.append(choice)
                progress = True
                if len(selected) >= target_size:
                    break
            if not progress:
                break

        if len(selected) < target_size:
            remaining = [
                trainer_id
                for group in available_groups
                for trainer_id in group
                if trainer_id not in selected
            ]
            if remaining:
                selected.extend(self._rng.sample(remaining, min(len(remaining), target_size - len(selected))))

        return selected[:target_size]
