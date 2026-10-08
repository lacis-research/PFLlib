"""Native PFLlib local training and resource profiles for FedLoad/HERAFL."""
import time
import numpy as np
import torch
from flcore.clients.clientbase import Client


class clientFedLoad(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.decay_gamma = args.learning_rate_decay_gamma
        self.profile = {}
        self.round_stats = {}
        self.pruning_coordinates = None
        self.retention = 1.0

    def set_submodel(self, model, coordinates, retention):
        self.model = model
        self.pruning_coordinates = coordinates
        self.retention = retention
        # Each round can have a different shape, so rebuild the optimizer.
        learning_rate = self.learning_rate
        if self.learning_rate_decay:
            learning_rate *= self.decay_gamma ** self.train_time_cost['num_rounds']
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def _synchronize(self):
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize(self.device)

    def train(self):
        trainloader = self.load_train_data()
        if not len(trainloader):
            raise ValueError(f'Client {self.id}: batch size exceeds training partition size.')
        self.model.train()
        self._synchronize()
        start = time.perf_counter()
        epochs = self.local_epochs
        if self.train_slow:
            epochs = np.random.randint(1, max(2, epochs // 2 + 1))
        for _ in range(epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                self.loss(self.model(x), y).backward()
                self.optimizer.step()
        self._synchronize()
        compute_time = time.perf_counter() - start
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += compute_time
        self.record_resources(compute_time)

    def record_resources(self, compute_time):
        size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        throughput = float(self.profile.get('throughput_mbps', 10.0))
        transfer_time = size * 8 / (throughput * 1_000_000)
        compute_energy = float(self.profile.get('e_cmp_w', self.profile.get('e_cmp', 0.5))) * compute_time
        upload_energy = float(self.profile.get('e_up_w', self.profile.get('e_up', 0.5))) * transfer_time
        download_energy = float(self.profile.get('e_down_w', self.profile.get('e_down', 0.5))) * transfer_time
        self.round_stats = {
            'compute_time': compute_time, 'download_time': transfer_time,
            'upload_time': transfer_time, 'round_time': compute_time + 2 * transfer_time,
            'compute_energy': compute_energy, 'upload_energy': upload_energy,
            'download_energy': download_energy,
            'energy_consumption': compute_energy + upload_energy + download_energy,
            'energy_source': 'profile_model', 'communication_source': 'profile_model',
            'download_bytes': size, 'upload_bytes': size,
            'total_communication_bytes': 2 * size, 'throughput_mbps': throughput,
        }

    def get_context(self, max_samples):
        clip = lambda value: float(np.clip(value, 0.0, 1.0))
        p, stats = self.profile, self.round_stats
        cpu = float(p.get('f_i', p.get('cpu_capacity', 0.5)))
        cost = float(p.get('c_r_i', p.get('compute_cost', 0.5)))
        if stats:
            cost = stats['compute_time'] / max(self.train_samples * self.local_epochs, 1)
            cost /= float(p.get('compute_time_scale_per_sample', 0.002))
        bandwidth = clip((float(p.get('throughput_mbps', 10.0)) - 1) / 74)
        return {
            'e_cmp': clip(p.get('e_cmp', 0.5)), 'e_up': clip(p.get('e_up', 0.5)),
            'e_down': clip(p.get('e_down', 0.5)), 'f_i': clip(cpu),
            'dataset_size': clip(self.train_samples / max(max_samples, 1)),
            'c_r_i': clip(cost), 'b_mean': bandwidth, 'cpu_capacity': clip(cpu),
            'compute_cost': clip(cost), 'bandwidth': bandwidth,
            'energy_efficiency': clip(p.get('energy_efficiency', 0.5)),
            'round_time_scale': float(p.get('round_time_scale', 60.0)),
            'round_time': clip(stats.get('round_time', 0) / float(p.get('round_time_scale', 60.0))),
            'energy_consumption': clip(stats.get('energy_consumption', 0) / float(p.get('energy_scale', 10.0))),
        }
