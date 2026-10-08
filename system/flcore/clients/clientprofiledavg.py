"""Measure FedAvg training without changing its original optimizer or update."""
import time
from flcore.clients.clientavg import clientAVG
from flcore.clients.clientfedload import clientFedLoad


class clientProfiledAVG(clientAVG, clientFedLoad):
    def train(self):
        self._synchronize()
        started = time.perf_counter()
        super().train()
        self._synchronize()
        self.record_resources(time.perf_counter() - started)
