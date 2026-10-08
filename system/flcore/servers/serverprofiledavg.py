"""Optional resource logging around the original FedAvg round lifecycle."""
import json
import os
import numpy as np
import torch
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverfedload import FedLoad
from flcore.clients.clientprofiledavg import clientProfiledAVG


class ProfiledFedAvg(FedAvg):
    client_class = clientProfiledAVG

    def __init__(self, args, times):
        super().__init__(args, times)
        self.round_reports = []
        profiles = {}
        if args.client_profiles:
            with open(args.client_profiles, encoding='utf-8') as stream:
                profiles = json.load(stream)
            if not isinstance(profiles, dict) or any(not isinstance(v, dict) for v in profiles.values()):
                raise ValueError('client_profiles must map IDs to profile objects.')
            unknown = set(profiles) - {str(c.id) for c in self.clients}
            if unknown:
                raise ValueError(f'Unknown client profile IDs: {sorted(unknown)}')
        self.max_samples = max(c.train_samples for c in self.clients)
        for client in self.clients:
            client.profile = profiles.get(str(client.id), {})
            FedLoad._validate_profile(client.profile)
        print('Resource logging: compute time measured; network and energy estimated from profiles.')

    def send_models(self):
        self.selection_contexts = {c.id: c.get_context(self.max_samples) for c in self.selected_clients}
        super().send_models()

    def receive_models(self):
        super().receive_models()
        clients = []
        for client in self.selected_clients:
            # Extra evaluation must not consume the RNG used by subsequent training.
            with torch.random.fork_rng(devices=[]):
                correct, count, _ = client.test_metrics()
            clients.append({'id': client.id, 'retention': 1.0,
                            'status': 'accepted' if client.id in self.uploaded_ids else 'rejected',
                            'accuracy': correct / max(count, 1), 'num_samples': client.train_samples,
                            'selection_context': self.selection_contexts[client.id],
                            'round_stats': dict(client.round_stats)})
        accuracies = [c['accuracy'] for c in clients if c['status'] == 'accepted']
        self.round_reports.append({'round': len(self.Budget), 'clients': clients,
                                   'mean_accuracy': float(np.mean(accuracies)) if accuracies else None})

    def save_results(self):
        super().save_results()
        path = os.path.join('../results', f'{self.dataset}_{self.algorithm}_{self.goal}_{self.times}_policy.json')
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump(self.round_reports, stream, indent=2, allow_nan=False)
