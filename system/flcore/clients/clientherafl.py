"""HERAFL training with optional post-training, deterministic semantic summaries."""
import time
from torch.utils.data import DataLoader
from flcore.clients.clientfedload import clientFedLoad
from flcore.heterogeneity.semantic import extract, wire_bytes


class clientHERAFL(clientFedLoad):
    def record_resources(self, compute_time):
        super().record_resources(compute_time)
        indices = getattr(self.model, 'semantic_input_indices', None)
        if indices is None:
            return
        # DNN needs its input-coordinate map at the receiver; server keeps it for upload.
        size = indices.numel() * indices.element_size()
        stats = self.round_stats
        seconds = size * 8 / (stats['throughput_mbps'] * 1_000_000)
        energy = seconds * self.profile.get('e_down_w', self.profile.get('e_down', .5))
        stats['coordinate_download_bytes'] = size
        stats['download_bytes'] += size
        stats['download_time'] += seconds
        stats['download_energy'] += energy
        stats['round_time'] += seconds
        stats['energy_consumption'] += energy
        stats['total_communication_bytes'] += size

    def train(self):
        super().train()
        if not getattr(self, 'semantic_collect', False):
            return
        self._synchronize()
        start = time.perf_counter()
        loader = DataLoader(self.load_train_data().dataset, batch_size=self.batch_size,
                            shuffle=False, drop_last=False)
        self.semantic_payload = extract(self.model, loader, self.num_classes, self.id,
                                        self.semantic_round, self.retention,
                                        self.semantic_snapshot, self.device)
        self._synchronize()
        duration = time.perf_counter() - start
        training_time = self.round_stats['compute_time']
        self.train_time_cost['total_cost'] += duration
        self.record_resources(training_time + duration)
        stats = self.round_stats
        down, up = wire_bytes(self.semantic_snapshot), wire_bytes(self.semantic_payload)
        stats.update(training_time=training_time, semantic_extraction_time=duration,
                     parameter_download_bytes=stats['download_bytes']-stats.get('coordinate_download_bytes', 0), parameter_upload_bytes=stats['upload_bytes'],
                     semantic_download_bytes=down, semantic_upload_bytes=up,
                     semantic_indicators=self.semantic_payload.indicators,
                     semantic_valid=self.semantic_payload.valid,
                     semantic_reference_round=self.semantic_snapshot.round,
                     semantic_reference_age=self.semantic_round-self.semantic_snapshot.round)
        for direction, size in (('download', down), ('upload', up)):
            seconds = size * 8 / (stats['throughput_mbps'] * 1_000_000)
            power = 'e_down' if direction == 'download' else 'e_up'
            energy = seconds * self.profile.get(power + '_w', self.profile.get(power, .5))
            stats[direction + '_bytes'] += size
            stats[direction + '_time'] += seconds
            stats[direction + '_energy'] += energy
            stats['round_time'] += seconds
            stats['energy_consumption'] += energy
            stats['total_communication_bytes'] += size
