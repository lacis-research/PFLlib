"""HERAFL is FedLoad plus optional semantic submodel selection."""
import numpy as np
from flcore.servers.serverfedload import FedLoad
from flcore.clients.clientherafl import clientHERAFL
from flcore.heterogeneity.linucb_fedload import LinUCBBandit


class HERAFL(FedLoad):
    client_class = clientHERAFL
    bandit_class = LinUCBBandit

    def __new__(cls, args, times):
        if cls is HERAFL and getattr(args, 'hera_communication', 'parameters') == 'prototypes':
            from flcore.servers.serverheraprototype import PrototypeHERAFL
            return object.__new__(PrototypeHERAFL)
        return object.__new__(cls)

    def __init__(self, args, times):
        from flcore.heterogeneity.semantic import SemanticLinUCB, Bank, wire_bytes
        from flcore.trainmodel.models import FedAvgCNN, DNN
        from flcore.trainmodel.vgg import VGG11, VGG16
        import math
        self.semantic_enabled = getattr(args, 'hera_semantic', False)
        if self.semantic_enabled:
            if getattr(args, 'hera_semantic_policy', 'linucb') not in ('linucb', 'heuristic', 'legacy'):
                raise ValueError('Unknown semantic policy.')
            if type(args.model) not in (FedAvgCNN, DNN, VGG11, VGG16):
                raise ValueError('--hera-semantic supports only FedAvgCNN, DNN, VGG11 and VGG16; MLR/MininetCNN require a separate feature mapping.')
            for key in ('hera_semantic_max_age', 'hera_semantic_warmup_rounds', 'hera_semantic_neighbors'):
                value = getattr(args, key, {'hera_semantic_max_age': 5, 'hera_semantic_warmup_rounds': 1, 'hera_semantic_neighbors': 1}[key])
                if not isinstance(value, int) or value < 0:
                    raise ValueError(key + ' must be a nonnegative integer.')
            for key, default in [('hera_semantic_reward_weight', 0.), ('hera_rpcl_weight', 0.), ('hera_cpdr_weight', 0.), ('hera_heuristic_weight', .6), ('hera_heuristic_cost_weight', .1)]:
                value = getattr(args, key, default)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(key + ' must be finite and nonnegative.')
            for key, default in [('hera_semantic_temperature', .05), ('hera_semantic_reward_time_scale', 60.), ('hera_semantic_memory_limit', None), ('hera_semantic_time_limit', None)]:
                value = getattr(args, key, default)
                if value is not None and (not math.isfinite(value) or value <= 0):
                    raise ValueError(key + ' must be finite and positive.')
            if getattr(args, 'hera_rpcl_weight', 0) or getattr(args, 'hera_cpdr_weight', 0):
                raise ValueError('RPCL/CPDR are optional future extensions; this implementation supports CE only.')
            features = getattr(args, 'hera_semantic_features', 'H,U,D,Q').split(',')
            if not features or len(set(features)) != len(features) or set(features)-set('HUDQ'):
                raise ValueError('hera_semantic_features must be a nonempty subset of H,U,D,Q.')
            self.semantic_features = features
            self.bandit_class = SemanticLinUCB
        super().__init__(args, times)
        if self.semantic_enabled:
            import copy
            from flcore.heterogeneity.pruning import prune_model
            self.bandit.legacy_policy = copy.deepcopy(self.bandit)
            self.bandit.enable(getattr(args, 'hera_semantic_policy', 'linucb'),
                               getattr(args, 'hera_heuristic_weight', .6), getattr(args, 'hera_heuristic_cost_weight', .1))
            self.semantic_max_age = getattr(args, 'hera_semantic_max_age', 5)
            self.semantic_warmup = getattr(args, 'hera_semantic_warmup_rounds', 1)
            self.semantic_reward_weight = getattr(args, 'hera_semantic_reward_weight', 0.)
            self.semantic_reward_time_scale = getattr(args, 'hera_semantic_reward_time_scale', 60.)
            self.semantic_memory_limit = getattr(args, 'hera_semantic_memory_limit', None)
            self.semantic_time_limit = getattr(args, 'hera_semantic_time_limit', None)
            self.semantic_collect = self.bandit.policy != 'legacy' or self.semantic_reward_weight > 0
            self.semantic_history = {}
            self.semantic_bank = Bank(dimension=self.global_model.fc.in_features)
            self.semantic_bank_bytes = wire_bytes(self.semantic_bank)
            total = sum(p.numel() for p in self.global_model.parameters())
            self.semantic_costs = {}
            for arm in self.bandit.pruning_rates:
                model, _ = prune_model(self.global_model, arm, device='cpu', preserve_representation=True)
                count = sum(p.numel() for p in model.parameters())
                self.semantic_costs[arm] = dict(fraction=count/total, bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                                                classifier_parameters=sum(p.numel() for p in model.fc.parameters()),
                                                coordinate_bytes=model.semantic_input_indices.numel()*8 if hasattr(model, 'semantic_input_indices') else 0)

    def prepare_contexts(self):
        contexts = super().prepare_contexts()
        if self.semantic_enabled:
            from flcore.heterogeneity.semantic import history_context
            for client_id, context in contexts.items():
                context.update(history_context(self.semantic_history.get(client_id), self.current_round,
                                               self.semantic_max_age, self.semantic_features))
            if getattr(self.args, 'hera_semantic_shuffle', False):
                ids = sorted(contexts)
                shuffled = np.random.default_rng(self.experiment_seed + self.current_round).permutation(ids)
                snapshots = {i: {k: v for k, v in contexts[i].items() if k.startswith('semantic_')} for i in ids}
                for target, source in zip(ids, shuffled):
                    contexts[target].update(snapshots[source])
        return contexts

    def update_policy(self, client, context, reward):
        super().update_policy(client, context, reward)
        if self.semantic_enabled:
            self.bandit.legacy_policy.update(client.retention, reward, context)

    def assign_submodel(self, client, context, pending_rates):
        if not self.semantic_enabled:
            return super().assign_submodel(client, context, pending_rates)
        from flcore.heterogeneity.pruning import prune_model
        from flcore.heterogeneity.semantic import Bank
        import copy
        costs = {}
        for arm, spec in self.semantic_costs.items():
            previous = client.round_stats
            if previous:
                previous_fraction = previous.get('parameter_fraction', 1.)
                compute = previous['compute_time'] * spec['fraction'] / max(previous_fraction, 1e-12)
            else:
                compute = client.train_samples * client.profile.get('compute_time_scale_per_sample', .002) * (client.local_epochs + int(self.semantic_collect)) * spec['fraction']
            # Bank is sent to each participant; use maximum class coverage for predicted upload.
            semantic_bytes = self.semantic_bank_bytes + self.num_classes * (self.global_model.fc.in_features * 4 + 24) + 1024 if self.semantic_collect else 0
            seconds = compute + (2*spec['bytes'] + spec['coordinate_bytes'] + semantic_bytes)*8/(context['throughput_mbps']*1e6)
            # Adam adds two moment tensors; neither estimate includes activations.
            memory = (5 if getattr(self.global_model, 'optimizer_kind', 'sgd') == 'adam' else 3)*spec['bytes']
            costs[arm] = dict(spec, predicted_time=seconds, predicted_memory=memory,
                              feasible=(self.semantic_memory_limit is None or memory <= self.semantic_memory_limit)
                              and (self.semantic_time_limit is None or seconds <= self.semantic_time_limit))
        fallback = self.current_round <= self.semantic_warmup or not context.get('semantic_available', 0)
        retention, info = self.bandit.choose_semantic(context, pending_rates, costs, fallback)
        info['fallback_reason'] = ('warmup' if self.current_round <= self.semantic_warmup else
                                   ('no_accepted_history' if client.id not in self.semantic_history else 'expired_or_unavailable')) if fallback else None
        if retention is None:
            return None, info
        model, coordinates = prune_model(self.global_model, retention, device='cpu', preserve_representation=True)
        client.set_submodel(model, coordinates, retention)
        client.semantic_collect = self.semantic_collect
        client.semantic_snapshot = copy.deepcopy(self.semantic_bank) if self.semantic_collect else Bank()
        client.semantic_round = self.current_round
        return retention, info

    def after_aggregation(self, responses, reports):
        if not self.semantic_enabled:
            return {}
        import time
        from flcore.heterogeneity.semantic import build_bank, wire_bytes
        start = time.perf_counter()
        payloads = [c.semantic_payload for c in responses] if self.semantic_collect else []
        self.semantic_bank = build_bank(payloads, self.semantic_bank, self.current_round)
        self.semantic_bank_bytes = wire_bytes(self.semantic_bank)
        for p in payloads:
            self.semantic_history[p.client_id] = p
        for report in reports:
            if report['retention'] is not None:
                spec = self.semantic_costs[report['retention']]
                report['round_stats'].update(parameter_fraction=spec['fraction'], classifier_parameters=spec['classifier_parameters'])
                self.clients[report['id']].round_stats.update(parameter_fraction=spec['fraction'])
                if self.semantic_collect:
                    report['semantic_payload'] = self.clients[report['id']].semantic_payload.metadata()
                client = self.clients[report['id']]
                for attribute in ('semantic_payload', 'semantic_snapshot'):
                    if hasattr(client, attribute):
                        delattr(client, attribute)
        return dict(semantic_bank_round=self.semantic_bank.round,
                    semantic_bank_age=self.current_round-self.semantic_bank.round,
                    semantic_classes=sorted(self.semantic_bank.prototypes),
                    semantic_contributors=self.semantic_bank.contributors,
                    semantic_aggregation_time=time.perf_counter()-start,
                    fallback_count=sum(bool(r['alpha_info']['fallback']) for r in reports))

    def calculate_reward(self, retention, accuracy, stats, time_reference):
        reward, components = super().calculate_reward(retention, accuracy, stats, time_reference)
        if self.semantic_enabled and self.semantic_reward_weight:
            values, valid = stats['semantic_indicators'], stats['semantic_valid']
            quality = values['Q'] if valid['Q'] else 0.
            cost = min(1., stats['round_time']/self.semantic_reward_time_scale)
            term = quality - cost
            components.update(control_reward=reward, semantic_quality=quality,
                              semantic_quality_valid=valid['Q'], normalized_cost=cost,
                              semantic_term=term, semantic_weight=self.semantic_reward_weight)
            reward = (reward + self.semantic_reward_weight*term)/(1+self.semantic_reward_weight)
        return reward, components
