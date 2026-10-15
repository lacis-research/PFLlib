"""FedLoad in PFLlib: LinUCB, structured submodels and masked aggregation."""
import copy
import json
import math
import os
import random
import time
from flcore.heterogeneity.profiles import profiles_for_algorithm
import numpy as np
import torch
from flcore.servers.serverbase import Server
from flcore.clients.clientfedload import clientFedLoad
from flcore.heterogeneity.linucb_fedload import LinUCBBandit
from flcore.heterogeneity.pruning import validate_model, prune_model, aggregate_submodels


class FedLoad(Server):
    client_class = clientFedLoad
    bandit_class = LinUCBBandit

    def __init__(self, args, times):
        super().__init__(args, times)
        validate_model(self.global_model)
        if not 0 < self.join_ratio <= 1:
            raise ValueError('join_ratio must be in (0, 1].')
        if not 0 <= self.client_drop_rate < 1:
            raise ValueError('client_drop_rate must be in [0, 1).')
        if args.num_new_clients or args.dlg_eval:
            raise ValueError('FedLoad/HERAFL do not yet support new-client fine tuning or DLG.')
        if args.local_epochs < 1 or args.eval_gap < 1 or args.global_rounds < 0:
            raise ValueError('local_epochs/eval_gap must be positive and global_rounds nonnegative.')
        if not args.pruning_arms or any(not math.isfinite(a) or not 0 < a <= 1 for a in args.pruning_arms):
            raise ValueError('pruning_arms must contain retention ratios in (0, 1].')
        if len(set(args.pruning_arms)) != len(args.pruning_arms):
            raise ValueError('pruning_arms must be distinct.')
        if args.bandit_alpha < 0 or not 0 <= args.bandit_beta <= 1:
            raise ValueError('bandit_alpha must be nonnegative; bandit_beta must be in [0, 1].')
        if len(args.reward_weights) != 3 or any(not math.isfinite(w) or w < 0 for w in args.reward_weights) or not np.isclose(sum(args.reward_weights), 1):
            raise ValueError('reward_weights must be three nonnegative values summing to one.')
        if args.pruning_round_timeout is not None and (not math.isfinite(args.pruning_round_timeout) or args.pruning_round_timeout <= 0):
            raise ValueError('pruning_round_timeout must be positive and finite.')
        self.wait_for_all = getattr(args, 'wait_for_all', False)
        if self.wait_for_all:
            self.client_drop_rate = 0.0
            self.time_select = False
        self.aligned_comparison = getattr(args, 'aligned_comparison', False)
        self.experiment_seed = args.policy_seed + times
        self.scenario = getattr(args, 'scenario', 'iid')
        self.base_round_timeout = math.inf if self.wait_for_all else (args.pruning_round_timeout or 180)
        self.reward_weights = args.reward_weights
        bandit_options = dict(alpha=args.bandit_alpha, beta=args.bandit_beta,
                              random_seed=self.experiment_seed,
                              alpha_strategy=getattr(args, 'bandit_alpha_strategy', 'binary'))
        if self.algorithm == 'HERAFL':
            bandit_options['capability_guidance'] = dict(
                enabled=True, strength=0.6, straggler_penalty=0.15,
                compute_weight=0.4, bandwidth_weight=0.25, energy_weight=0.15,
                straggler_weight=0.2, static_bias=0.75, dynamic_penalty=0.25,
                high_capacity_floor=0.65)
            bandit_options['retention_floors'] = getattr(args, 'hera_retention_floors', (0.70, 0.85))
            bandit_options['capacity_thresholds'] = getattr(args, 'hera_capacity_thresholds', (0.40, 0.70))
        self.bandit = self.bandit_class(args.pruning_arms, **bandit_options)
        self.set_slow_clients()
        self.set_clients(self.client_class)
        profiles = profiles_for_algorithm(self.num_clients, args.policy_seed, self.aligned_comparison or self.algorithm == "HERAFL",
                                          preset=getattr(args, "resource_preset", "mininet"))
        if args.client_profiles:
            with open(args.client_profiles, encoding='utf-8') as stream:
                overrides = json.load(stream)
            if not isinstance(overrides, dict) or any(not isinstance(v, dict) for v in overrides.values()):
                raise ValueError('client_profiles must map client IDs to profile objects.')
            unknown = set(overrides) - {str(c.id) for c in self.clients}
            if unknown:
                raise ValueError(f'Unknown client profile IDs: {sorted(unknown)}')
            for client_id, profile in overrides.items():
                profiles[client_id].update(profile)
        self.max_samples = max(c.train_samples for c in self.clients)
        self.initial_contexts, self.contexts = {}, {}
        for client in self.clients:
            client.model = self.global_model
            client.configure_profile(profiles.get(str(client.id), {}))
            self._validate_profile(client.profile)
            self.contexts[client.id] = client.get_context(self.max_samples)
            self.initial_contexts[client.id] = dict(self.contexts[client.id])
        self.arm_energy_history = {a: [] for a in self.bandit.pruning_rates}
        self.round_reports = []
        self.Budget = []
        self.current_round = 0
        print(f'\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}')
        print('Finished creating server and clients.')
        print('Network time and energy are estimates from profiles; compute time is measured.')

    @staticmethod
    def _validate_profile(profile):
        positive = ('throughput_mbps', 'round_time_scale', 'energy_scale', 'compute_time_scale_per_sample')
        normalized = ('bandwidth', 'uplink_energy', 'downlink_energy', 'e_cmp', 'e_up', 'e_down', 'f_i', 'cpu_capacity', 'c_r_i', 'compute_cost', 'energy_efficiency')
        allowed = set(positive + normalized + ('e_cmp_w', 'e_up_w', 'e_down_w', 'bandwidth_window', 'max_throughput_mbps', 'sampled_throughput_mbps', 'cpu_frequency_ghz', 'cpu_frequency_min_ghz', 'cpu_frequency_max_ghz', 'compute_cycles_per_sample', 'compute_cycles_min', 'compute_cycles_max', 'random_seed'))
        if set(profile) - allowed:
            raise ValueError(f'Unknown profile fields: {sorted(set(profile) - allowed)}')
        for key, value in profile.items():
            value = float(value)
            if not math.isfinite(value) or value < 0 or (key in positive and value <= 0) or (key in normalized and value > 1):
                raise ValueError(f'Invalid profile value for {key}: {value}')

    def prepare_contexts(self):
        return {c.id: dict(self.contexts[c.id]) for c in self.selected_clients}

    def round_timeout(self, contexts):
        return self.base_round_timeout

    @staticmethod
    def distance_utility(reference, observed):
        if reference <= 0 or observed < 0:
            return 1.0
        return max(0.0, min(1.0, 1 - abs(observed - reference) / reference))

    def calculate_reward(self, retention, accuracy, stats, time_reference):
        history = self.arm_energy_history[retention]
        energy_reference = float(np.mean(history)) if history else stats['energy_consumption']
        components = [float(np.clip(accuracy, 0, 1)),
                      self.distance_utility(energy_reference, stats['energy_consumption']),
                      self.distance_utility(time_reference, stats['round_time'])]
        return float(np.dot(self.reward_weights, components)), {
            'Uacc': components[0], 'UE': components[1], 'UT': components[2],
            'energy_reference': energy_reference, 'time_reference': time_reference}

    def update_policy(self, client, context, reward):
        self.bandit.update(client.retention, reward, context)

    def train(self):
        for round_index in range(1, self.global_rounds + 1):
            self.current_round = round_index
            print(f'\n-------------Round number: {round_index}-------------', flush=True)
            started = time.perf_counter()
            self.selected_clients = self.select_clients()
            contexts = self.prepare_contexts()
            timeout = self.round_timeout(contexts)
            pending_rates = []
            assignments = {}
            for client in self.selected_clients:
                retention, info = self.bandit.choose_pruning_rate(contexts[client.id], pending_rates)
                pending_rates.append(retention)
                submodel, coordinates = prune_model(self.global_model, retention, device='cpu')
                client.set_submodel(submodel, coordinates, retention)
                assignments[client.id] = info
                client.send_time_cost['num_rounds'] += 1
            # Do not retain the last assignment through the following evaluation.
            del submodel, coordinates
            active_count = int((1 - self.client_drop_rate) * len(self.selected_clients))
            active_ids = {c.id for c in self.selected_clients} if active_count == len(self.selected_clients) else {c.id for c in (random.Random(self.experiment_seed + round_index) if self.aligned_comparison else random).sample(self.selected_clients, active_count)}
            responses, reports = [], []
            for client in self.selected_clients:
                if self.aligned_comparison:
                    seed = self.experiment_seed + round_index * self.num_clients + client.id
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    client.system_rng = np.random.default_rng(np.random.SeedSequence([self.experiment_seed, round_index, client.id]))
                client.train()
                stats = client.round_stats
                client.send_time_cost['total_cost'] += stats['download_time'] + stats['upload_time']
                self.contexts[client.id] = client.get_context(self.max_samples)
                accepted = client.id in active_ids and stats['round_time'] <= timeout
                if self.time_select:
                    accepted = accepted and stats['round_time'] <= self.time_threthold
                correct, samples = client.local_accuracy_metrics()
                response_context = dict(self.contexts[client.id])
                response_context.update({k: v for k, v in contexts[client.id].items() if k.startswith('static_') or k in ('predicted_round_time', 'straggler_risk', 'straggler_threshold')})
                report = {'context': response_context, 'id': client.id, 'retention': client.retention,
                          'status': 'accepted' if accepted else ('dropped' if client.id not in active_ids else 'timeout'),
                          'accuracy': correct / max(samples, 1), 'num_samples': client.train_samples,
                          'selection_context': contexts[client.id], 'alpha_info': assignments[client.id],
                          'round_stats': dict(stats)}
                client.offload_submodel()
                reports.append(report)
                if accepted:
                    responses.append(client)
            time_reference = float(np.mean([c.round_stats['round_time'] for c in responses])) if responses else 0.0
            updates = []
            pending_energy = []
            for client in responses:
                report = next(r for r in reports if r['id'] == client.id)
                reward, components = self.calculate_reward(client.retention, report['accuracy'], client.round_stats, time_reference)
                report.update(reward=reward, reward_components=components)
                # FedLoad retains its response-context update; HERAFL learns from the decision context.
                learning_context = dict(self.contexts[client.id])
                for key, value in contexts[client.id].items():
                    if key.startswith('static_') or key in ('predicted_round_time', 'straggler_risk', 'straggler_threshold'):
                        learning_context[key] = value
                if self.algorithm == 'HERAFL':
                    learning_context = dict(contexts[client.id])
                report['learning_context'] = dict(learning_context)
                self.update_policy(client, learning_context, reward)
                pending_energy.append((client.retention, client.round_stats['energy_consumption']))
                updates.append((client.model, client.pruning_coordinates, client.train_samples))
            for retention, energy in pending_energy:
                self.arm_energy_history[retention].append(energy)
            self.uploaded_ids = [c.id for c in responses]
            self.global_model = aggregate_submodels(self.global_model, updates)
            if round_index % self.eval_gap == 0:
                for client in self.clients:
                    client.model = self.global_model
                self.evaluate()
            self.round_reports.append({'global_accuracy': self.rs_test_acc[-1] if round_index % self.eval_gap == 0 else None, 'round': round_index, 'round_timeout': None if math.isinf(timeout) else timeout,
                                       'clients': reports})
            self.Budget.append(time.perf_counter() - started)
            print('Retentions:', {c.id: c.retention for c in self.selected_clients})
            print('Accepted updates:', len(responses), '| time cost:', self.Budget[-1])
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        print('\nBest accuracy:', max(self.rs_test_acc, default=0))
        print('Average time cost per round:', float(np.mean(self.Budget[1:] or self.Budget)) if self.Budget else 0.0)
        self.save_results()
        self.save_global_model()
        self.save_policy_results()

    def save_policy_results(self):
        from flcore.heterogeneity.results import save_policy_results
        save_policy_results(self)

    def select_clients(self):
        count = max(1, int(self.num_clients * self.join_ratio))
        if self.random_join_ratio:
            count = int(np.random.default_rng(self.experiment_seed + self.current_round).integers(count, self.num_clients + 1)) if self.aligned_comparison else int(np.random.randint(count, self.num_clients + 1))
        rng = random.Random(self.experiment_seed + self.current_round) if self.aligned_comparison else random
        return rng.sample(self.clients, count)


class FullModelPolicy:
    """Fixed full-model baseline under the same experiment lifecycle."""
    def __init__(self, pruning_rates, **kwargs):
        self.pruning_rates = [1.0]

    def choose_pruning_rate(self, context, pruning_reference=None):
        return 1.0, {'selection_mode': 'full_model'}

    def update(self, chosen_rate, reward, context):
        pass


class ComparisonFedAvg(FedLoad):
    """FedAvg with full local models and the shared synchronous round protocol."""
    bandit_class = FullModelPolicy
