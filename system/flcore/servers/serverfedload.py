"""FedLoad in PFLlib: LinUCB, structured submodels and masked aggregation."""
import copy
import json
import math
import os
import random
import time
import numpy as np
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
        if self.num_join_clients < 1:
            raise ValueError('join_ratio must select at least one client.')
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
        if not math.isfinite(args.pruning_round_timeout) or args.pruning_round_timeout <= 0:
            raise ValueError('pruning_round_timeout must be positive and finite.')
        self.base_round_timeout = args.pruning_round_timeout
        self.reward_weights = args.reward_weights
        self.bandit = self.bandit_class(args.pruning_arms, alpha=args.bandit_alpha,
                                       beta=args.bandit_beta, random_seed=args.policy_seed + times)
        self.set_slow_clients()
        self.set_clients(self.client_class)
        profiles = {}
        if args.client_profiles:
            with open(args.client_profiles, encoding='utf-8') as stream:
                profiles = json.load(stream)
            if not isinstance(profiles, dict) or any(not isinstance(v, dict) for v in profiles.values()):
                raise ValueError('client_profiles must map client IDs to profile objects.')
            unknown = set(profiles) - {str(c.id) for c in self.clients}
            if unknown:
                raise ValueError(f'Unknown client profile IDs: {sorted(unknown)}')
        self.max_samples = max(c.train_samples for c in self.clients)
        self.initial_contexts, self.contexts = {}, {}
        for client in self.clients:
            client.profile = profiles.get(str(client.id), {})
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
        normalized = ('e_cmp', 'e_up', 'e_down', 'f_i', 'cpu_capacity', 'c_r_i', 'compute_cost', 'energy_efficiency')
        allowed = set(positive + normalized + ('e_cmp_w', 'e_up_w', 'e_down_w'))
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
        for round_index in range(self.global_rounds + 1):
            self.current_round = round_index
            started = time.perf_counter()
            self.selected_clients = self.select_clients()
            # Evaluate the complete global model using PFLlib's original metrics.
            for client in self.clients:
                client.model = copy.deepcopy(self.global_model)
            if round_index % self.eval_gap == 0:
                print(f'\n-------------Round number: {round_index}-------------')
                self.evaluate()
            contexts = self.prepare_contexts()
            timeout = self.round_timeout(contexts)
            pending_rates = []
            assignments = {}
            for client in self.selected_clients:
                retention, info = self.bandit.choose_pruning_rate(contexts[client.id], pending_rates)
                pending_rates.append(retention)
                submodel, coordinates = prune_model(self.global_model, retention)
                client.set_submodel(submodel, coordinates, retention)
                assignments[client.id] = info
                client.send_time_cost['num_rounds'] += 1
            active_count = int((1 - self.client_drop_rate) * len(self.selected_clients))
            active_ids = {c.id for c in random.sample(self.selected_clients, active_count)}
            responses, reports = [], []
            for client in self.selected_clients:
                client.train()
                stats = client.round_stats
                client.send_time_cost['total_cost'] += stats['download_time'] + stats['upload_time']
                self.contexts[client.id] = client.get_context(self.max_samples)
                accepted = client.id in active_ids and stats['round_time'] <= timeout
                if self.time_select:
                    accepted = accepted and stats['round_time'] <= self.time_threthold
                correct, samples, _ = client.test_metrics()
                report = {'id': client.id, 'retention': client.retention,
                          'status': 'accepted' if accepted else ('dropped' if client.id not in active_ids else 'timeout'),
                          'accuracy': correct / max(samples, 1), 'num_samples': client.train_samples,
                          'selection_context': contexts[client.id], 'alpha_info': assignments[client.id],
                          'round_stats': dict(stats)}
                reports.append(report)
                if accepted:
                    responses.append(client)
            time_reference = float(np.mean([c.round_stats['round_time'] for c in responses])) if responses else 0.0
            updates = []
            for client in responses:
                report = next(r for r in reports if r['id'] == client.id)
                reward, components = self.calculate_reward(client.retention, report['accuracy'], client.round_stats, time_reference)
                report.update(reward=reward, reward_components=components)
                # Source controllers learn from the response context, enriched for HERAFL.
                learning_context = dict(self.contexts[client.id])
                for key, value in contexts[client.id].items():
                    if key.startswith('static_') or key in ('predicted_round_time', 'straggler_risk', 'straggler_threshold'):
                        learning_context[key] = value
                self.update_policy(client, learning_context, reward)
                self.arm_energy_history[client.retention].append(client.round_stats['energy_consumption'])
                updates.append((client.model, client.pruning_coordinates, client.train_samples))
            self.uploaded_ids = [c.id for c in responses]
            self.global_model = aggregate_submodels(self.global_model, updates)
            self.round_reports.append({'round': round_index, 'round_timeout': None if math.isinf(timeout) else timeout,
                                       'clients': reports})
            self.Budget.append(time.perf_counter() - started)
            print('Retentions:', {c.id: c.retention for c in self.selected_clients})
            print('Accepted updates:', len(responses), '| time cost:', self.Budget[-1])
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        print('\nBest accuracy:', max(self.rs_test_acc))
        print('Average time cost per round:', float(np.mean(self.Budget[1:] or self.Budget)))
        self.save_results()
        self.save_global_model()
        path = os.path.join('../results', f'{self.dataset}_{self.algorithm}_{self.goal}_{self.times}_policy.json')
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump(self.round_reports, stream, indent=2, allow_nan=False)
