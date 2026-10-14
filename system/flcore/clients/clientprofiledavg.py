"""Measure FedAvg training without changing its original optimizer or update."""
import time
import torch
from flcore.clients.clientavg import clientAVG
from flcore.clients.clientfedload import clientFedLoad


class clientProfiledAVG(clientAVG, clientFedLoad):
    def _move_optimizer_state(self, device):
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != 'step':
                    state[key] = value.to(device)

    def set_parameters(self, model):
        with torch.no_grad():
            for source, target in zip(model.parameters(), self.model.parameters()):
                target.copy_(source.to(target.device))

    def train(self):
        self.model.to(self.device)
        self._move_optimizer_state(self.device)
        self.sample_system_state()
        self._synchronize()
        started = time.perf_counter()
        super().train()
        self._synchronize()
        self.record_resources(time.perf_counter() - started)
        self.model.cpu()
        self._move_optimizer_state('cpu')

    def test_metrics(self):
        original_device = next(self.model.parameters()).device
        self.model.to(self.device)
        try:
            return super().test_metrics()
        finally:
            self.model.to(original_device)

    def train_metrics(self):
        original_device = next(self.model.parameters()).device
        self.model.to(self.device)
        try:
            return super().train_metrics()
        finally:
            self.model.to(original_device)
