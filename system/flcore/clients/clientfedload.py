"""Native PFLlib local training and resource profiles for FedLoad/HERAFL."""
import time
from torch.utils.data import DataLoader
import numpy as np
import torch
from flcore.clients.clientbase import Client


class clientFedLoad(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.validation_fraction = getattr(args, 'reward_validation_fraction', 0.0)
        self.validation_seed = getattr(args, 'policy_seed', 42)
        if not np.isfinite(self.validation_fraction) or not 0 <= self.validation_fraction < 1:
            raise ValueError('reward_validation_fraction must be in [0, 1).')
        self.decay_gamma = args.learning_rate_decay_gamma
        self.profile = {}
        self.system_rng = np.random.default_rng(4200 + id)
        self.current_system_state = {}
        self.bandwidth_history = []
        self.round_stats = {}
        self.pruning_coordinates = None
        self.retention = 1.0
        if self.algorithm in ('FedLoad', 'HERAFL'):
            self.optimizer = None
            self.learning_rate_scheduler = None

    def _split_train_data(self, batch_size=None):
        loader = super().load_train_data(batch_size)
        if not self.validation_fraction:
            return loader, None
        from torch.utils.data import Subset
        order = np.random.default_rng(self.validation_seed + self.id).permutation(len(loader.dataset))
        count = max(1, int(len(order) * self.validation_fraction))
        if count >= len(order):
            raise ValueError('Validation split requires at least two training examples.')
        validation = Subset(loader.dataset, order[:count].tolist())
        return DataLoader(Subset(loader.dataset, order[count:].tolist()),
                          batch_size=batch_size or self.batch_size, shuffle=True, drop_last=False), validation

    def load_train_data(self, batch_size=None):
        # Keep validation local: a Subset retains the entire source partition.
        return self._split_train_data(batch_size)[0]

    def set_submodel(self, model, coordinates, retention):
        self.model = model
        self.pruning_coordinates = coordinates
        self.retention = retention
        # Each round can have a different shape, so rebuild the optimizer.
        learning_rate = self.learning_rate
        if self.learning_rate_decay:
            learning_rate *= self.decay_gamma ** self.train_time_cost['num_rounds']
        if getattr(self.model, 'optimizer_kind', 'sgd') == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-7)
        elif getattr(self, 'reference_training', False):
            if self.model.augment:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-7)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                                  momentum=0.9)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def offload_submodel(self):
        """Release round-local optimizer state and keep received updates on CPU."""
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.model.cpu()
        self.pruning_coordinates = {name: tuple(axis.cpu() for axis in axes)
                                    for name, axes in self.pruning_coordinates.items()}

    def _synchronize(self):
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize(self.device)

    def train(self):
        self.model.to(self.device)
        self.sample_system_state()
        trainloader = DataLoader(self.load_train_data().dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
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
                self.training_loss(x, y).backward()
                if self.reference_training and not self.model.augment:
                    # Keras weight_decay is decoupled from the momentum gradient.
                    with torch.no_grad():
                        for parameter in self.model.parameters():
                            parameter.mul_(1 - self.optimizer.param_groups[0]['lr'] * 5e-4)
                self.optimizer.step()
        self._synchronize()
        compute_time = time.perf_counter() - start
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += compute_time
        self.record_resources(compute_time)

    def training_loss(self, x, y):
        return self.loss(self.model(x), y)

    def record_resources(self, compute_time):
        size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        throughput = float(self.current_system_state.get('throughput_mbps', self.profile.get('throughput_mbps', 10.0)))
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
            'parameter_download_bytes': size, 'parameter_upload_bytes': size,
            'total_communication_bytes': 2 * size, 'throughput_mbps': throughput,
        }

    def configure_profile(self, profile):
        self.profile = profile
        self.client_profile = profile
        self.system_rng = np.random.default_rng(int(profile.get('random_seed', 4200 + self.id)))
        self.current_system_state = {}
        self.bandwidth_history = []
        self.reference_training = getattr(self, 'reference_training', False)

    def local_accuracy_metrics(self):
        """Evaluate the client's native PFLlib test partition."""
        if self.validation_fraction:
            _, validation = self._split_train_data()
            loader = DataLoader(validation, batch_size=self.batch_size, shuffle=False)
        else:
            loader = self.load_test_data()
        correct = samples = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                correct += int((self.model(x).argmax(1) == y).sum())
                samples += len(y)
        return correct, samples

    def _normalize(self, value, default=0.0):
        return default if value is None else float(np.clip(value, 0, 1))

    def _normalize_range(self, value, lower, upper, default=0.0):
        return default if value is None or upper <= lower else self._normalize((value - lower) / (upper - lower))

    def sample_system_state(self):
        self.round_index = getattr(self, "round_index", 0) + 1
        max_throughput = float(self.client_profile.get("max_throughput_mbps", 10.0))
        throughput_mbps = float(self.system_rng.uniform(1.0, max_throughput))

        base_frequency = float(self.client_profile.get("cpu_frequency_ghz", 1.5))
        frequency_floor = float(self.client_profile.get("cpu_frequency_min_ghz", 1.0))
        frequency_ceiling = float(self.client_profile.get("cpu_frequency_max_ghz", 2.0))
        frequency_low = max(frequency_floor, base_frequency * 0.85)
        frequency_high = min(frequency_ceiling, base_frequency * 1.05)
        cpu_frequency_ghz = float(self.system_rng.uniform(frequency_low, max(frequency_low + 1e-6, frequency_high)))

        base_cycles = float(self.client_profile.get("compute_cycles_per_sample", 15.0))
        cycles_floor = float(self.client_profile.get("compute_cycles_min", 5.0))
        cycles_ceiling = float(self.client_profile.get("compute_cycles_max", 30.0))
        cycles_low = max(cycles_floor, base_cycles * 0.85)
        cycles_high = min(cycles_ceiling, base_cycles * 1.15)
        compute_cycles = float(self.system_rng.uniform(cycles_low, max(cycles_low + 1e-6, cycles_high)))

        throughput_norm = self._normalize((throughput_mbps - 1.0) / (75.0 - 1.0))
        cpu_frequency_norm = self._normalize_range(
            cpu_frequency_ghz,
            frequency_floor,
            frequency_ceiling,
            default=self._normalize(float(self.client_profile.get("f_i", self.client_profile.get("cpu_capacity", 0.5))))
        )
        compute_cycles_norm = self._normalize_range(
            compute_cycles,
            cycles_floor,
            cycles_ceiling,
            default=self._normalize(float(self.client_profile.get("c_r_i", self.client_profile.get("compute_cost", 0.5))))
        )

        self.current_system_state = {
            "throughput_mbps": throughput_mbps,
            "throughput_norm": throughput_norm,
            "cpu_frequency_ghz": cpu_frequency_ghz,
            "cpu_frequency_norm": cpu_frequency_norm,
            "compute_cycles_per_sample": compute_cycles,
            "compute_cycles_norm": compute_cycles_norm
        }
        return dict(self.current_system_state)

    def get_context(self, max_samples):
        p, stats, state = self.profile, self.round_stats, self.current_system_state
        clip = self._normalize
        throughput = state.get('throughput_mbps', p.get('sampled_throughput_mbps', p.get('throughput_mbps', 10.0)))
        self.bandwidth_history.append(clip((throughput - 1) / 74))
        self.bandwidth_history = self.bandwidth_history[-int(p.get('bandwidth_window', 3)):]
        cpu = self._normalize_range(state.get('cpu_frequency_ghz', p.get('cpu_frequency_ghz', 1.5)),
                                   p.get('cpu_frequency_min_ghz', 1.0), p.get('cpu_frequency_max_ghz', 2.0),
                                   p.get('f_i', 0.5))
        cost = self._normalize_range(state.get('compute_cycles_per_sample', p.get('compute_cycles_per_sample', 15.0)),
                                    p.get('compute_cycles_min', 5.0), p.get('compute_cycles_max', 30.0), p.get('c_r_i', 0.5))
        if stats:
            cost = clip(stats['compute_time'] / max(self.train_samples, 1) / p.get('compute_time_scale_per_sample', 0.002))
        bandwidth = float(np.mean(self.bandwidth_history))
        return dict(e_cmp=clip(p.get('e_cmp', 0.5)), e_up=clip(p.get('e_up', 0.5)),
                    e_down=clip(p.get('e_down', 0.5)), f_i=cpu, cpu_capacity=cpu,
                    dataset_size=clip(self.train_samples / max(max_samples, 1)), c_r_i=cost,
                    compute_cost=cost, bandwidth=bandwidth, b_mean=bandwidth,
                    energy_efficiency=clip(p.get('energy_efficiency', 0.5)),
                    throughput_mbps=throughput,
                    round_time=clip(stats.get('round_time', 0) / p.get('round_time_scale', 60.0)),
                    energy_consumption=clip(stats.get('energy_consumption', 0) / p.get('energy_scale', 10.0)))
