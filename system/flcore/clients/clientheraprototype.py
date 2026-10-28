"""Persistent local pruning; only prototype/telemetry messages leave this client."""
import copy
from contextlib import contextmanager
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from flcore.clients.clientfedload import clientFedLoad
from flcore.heterogeneity.pruning import prune_model
from flcore.heterogeneity.semantic import extract
from flcore.heterogeneity.prototype_protocol import (
    PROTOCOL, encode, decode, payload_dict, reference_bank, semantic_losses,
)


def indices(axes):
    return (axes[0][:, None], axes[1][None, :]) if len(axes) == 2 else axes


class clientHERAPrototype(clientFedLoad):
    def configure_storage(self, path=None):
        self.state_path = path

    def offload_state(self):
        """Atomically persist heavy tensors; metadata remains available to policy."""
        if getattr(self, 'state_path', None) is None:
            return
        temporary = self.state_path.with_suffix('.tmp')
        try:
            torch.save(dict(master=self.master, model=self.model, moments=self.moments,
                            coordinates=self.pruning_coordinates), temporary)
            temporary.replace(self.state_path)
        finally:
            temporary.unlink(missing_ok=True)
        self.release_state()

    def release_state(self):
        self.master = self.model = None
        self.moments = {}
        self.pruning_coordinates = None
        self.optimizer = self.learning_rate_scheduler = None

    @contextmanager
    def resident(self, write=True):
        """Load only the current client's tensors, without changing RNG state."""
        if getattr(self, 'state_path', None) is None:
            yield
            return
        state = torch.load(self.state_path, map_location='cpu', weights_only=False)
        self.master, self.model = state['master'], state['model']
        self.moments, self.pruning_coordinates = state['moments'], state['coordinates']
        del state
        try:
            yield
            if write:
                self.offload_state()
        finally:
            self.release_state()

    def initialize_local(self, template, initialization, message=None):
        # The simulator may decode the identical broadcast once for all clients.
        message = decode(initialization) if message is None else message
        if message['protocol'] != PROTOCOL or message['kind'] != 'initialization':
            raise ValueError('Invalid initialization message.')
        self.master = copy.deepcopy(template).cpu()
        self.master.load_state_dict(message['state'])
        self.moments = {}
        self.model = self.master
        self.pruning_coordinates = None
        self.trained_rounds = 0
        self.prototype_options = None
        self.initialization_bytes = len(initialization)
        self.local_config = message.get('local_config')
        if self.local_config:
            from flcore.heterogeneity.linucb_fedload import LinUCBBandit
            cfg = self.local_config
            self.local_bandit = LinUCBBandit(cfg['arms'], alpha=cfg['alpha'], beta=cfg['beta'], random_seed=cfg['seed']+self.id, alpha_strategy=cfg['alpha_strategy'])
            self.local_energy = {a: [] for a in cfg['arms']}

    def install_local(self, retention):
        """Dormant weights/moments stay in CPU master; new moments start at zero.

        Adam uses its usual scalar step per parameter, retained across shape
        changes. Dormant entries are frozen (not decayed during inactivity).
        Supported models have no BatchNorm/running-stat buffers.
        """
        start = time.perf_counter()
        model, coordinates = prune_model(self.master, retention, device='cpu', preserve_representation=True)
        model.to(self.device)
        self.set_submodel(model, coordinates, retention)
        for name, parameter in model.named_parameters():
            saved = self.moments.get(name)
            if saved:
                self.optimizer.state[parameter] = {
                    key: (value[indices(coordinates[name])].clone().to(self.device) if value.ndim else value.clone())
                    for key, value in saved.items()}
        self._synchronize()
        return time.perf_counter()-start

    def commit_local(self):
        """Local scatter, never a server aggregation. Keep updates even on timeout."""
        self.model.cpu()
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                ix = indices(self.pruning_coordinates[name])
                self.master.get_parameter(name)[ix] = parameter
                state = self.optimizer.state.get(parameter, {})
                if state:
                    destination = self.moments.setdefault(name, {})
                    for key, value in state.items():
                        if value is None:
                            continue
                        value = value.detach().cpu()
                        if value.ndim == 0:
                            destination[key] = value.clone()
                        else:
                            if key not in destination:
                                destination[key] = torch.zeros_like(self.master.get_parameter(name))
                            destination[key][ix] = value
        self.model.zero_grad(set_to_none=True)
        self.optimizer = None
        self.learning_rate_scheduler = None

    def local_state_bytes(self):
        total = sum(p.numel()*p.element_size() for p in self.master.parameters())
        if self.model is not self.master:
            total += sum(p.numel()*p.element_size() for p in self.model.parameters())
        total += sum(v.numel()*v.element_size() for state in self.moments.values() for v in state.values())
        return total

    def exchange(self, request):
        message = decode(request)
        allowed = {'protocol', 'kind', 'client_id', 'round', 'retention', 'reference', 'options', 'seed', 'system_seed'}
        if set(message) != allowed or message['protocol'] != PROTOCOL or message['kind'] != 'train' or message['client_id'] != self.id:
            raise ValueError('Invalid prototype training request.')
        options, reference = message['options'], message['reference']
        if reference['dimension'] != self.master.fc.in_features:
            raise ValueError('Reference dimension mismatch.')
        self.prototype_options = options
        adaptation = self.install_local(message['retention'])
        if message.get('seed') is not None:
            seed = message['seed']
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.system_rng = np.random.default_rng(np.random.SeedSequence(message['system_seed']))
        self.sample_system_state()
        loader = DataLoader(self.load_train_data().dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.model.train()
        self._synchronize()
        start = time.perf_counter()
        for _ in range(self.local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits, features = self.model.forward_with_features(x)
                loss = self.loss(logits, y)
                if options['alignment'] or options['rpcl'] or options['cpdr']:
                    align, contrast = semantic_losses(features, y, reference, options['temperature'])
                    loss = loss + (options['alignment']+options['cpdr'])*align + options['rpcl']*contrast
                loss.backward()
                self.optimizer.step()
        self._synchronize()
        training_time = time.perf_counter()-start
        self.train_time_cost['num_rounds'] += 1
        self.trained_rounds += 1
        start = time.perf_counter()
        if self.local_config:
            from flcore.heterogeneity.semantic import Payload
            payload = Payload(self.id, message['round'], self.retention, self.master.fc.in_features, {}, {}, dict.fromkeys('HUDQ', 0.), dict.fromkeys('HUDQ', False), 0., 0)
        else:
            payload = extract(self.model, DataLoader(loader.dataset, batch_size=self.batch_size, shuffle=False),
                              self.num_classes, self.id, message['round'], self.retention,
                              reference_bank(reference), self.device)
        self._synchronize()
        extraction_time = time.perf_counter()-start
        start = time.perf_counter()
        correct, samples = self.local_accuracy_metrics()  # validation is required by server
        self._synchronize()
        validation_time = time.perf_counter()-start
        active_bytes = sum(p.numel()*p.element_size() for p in self.model.parameters())
        master_bytes = sum(p.numel()*p.element_size() for p in self.master.parameters())
        start = time.perf_counter()
        self.commit_local()
        adaptation += time.perf_counter()-start
        telemetry = dict(training_time=training_time, semantic_extraction_time=extraction_time,
                         adaptation_time=adaptation, validation_time=validation_time,
                         validation_correct=correct, validation_samples=samples,
                         system_state=self.current_system_state, parameter_fraction=active_bytes/master_bytes,
                         active_parameter_bytes=active_bytes, local_state_bytes=self.local_state_bytes())
        self.train_time_cost['total_cost'] += training_time+extraction_time+adaptation+validation_time
        return encode(dict(protocol=PROTOCOL, kind='response', payload=payload_dict(payload), telemetry=telemetry))

    def local_round(self, round_index):
        """Independent baseline. Request/response objects below are local traces,
        not messages; the simulator reads them only for offline reporting.
        Participant schedule and round clock are public benchmark configuration.
        """
        from flcore.heterogeneity.prototype_protocol import empty_reference, resource_stats
        cfg = self.local_config
        context = self.get_context(cfg['max_samples'])
        arm, info = self.local_bandit.choose_pruning_rate(context, [])
        request = encode(dict(protocol=PROTOCOL, kind='train', client_id=self.id, round=round_index,
                              retention=arm, reference=empty_reference(self.master.fc.in_features),
                              options=dict(alignment=0., rpcl=0., cpdr=0., temperature=.05),
                              seed=cfg['seed']+round_index*cfg['num_clients']+self.id,
                              system_seed=[cfg['seed'], round_index, self.id]))
        response = self.exchange(request)
        telemetry = decode(response)['telemetry']
        stats = resource_stats(telemetry, 0, 0, self.profile)
        history = self.local_energy[arm]
        energy_reference = float(np.mean(history)) if history else stats['energy_consumption']
        energy_utility = max(0., 1-abs(stats['energy_consumption']-energy_reference)/max(energy_reference, 1e-12))
        time_reference = self.round_stats.get('round_time', stats['round_time'])
        time_utility = max(0., 1-abs(stats['round_time']-time_reference)/max(time_reference, 1e-12))
        reward = float(np.dot(cfg['reward_weights'], [telemetry['validation_correct']/max(telemetry['validation_samples'], 1), energy_utility, time_utility]))
        self.local_bandit.update(arm, reward, context)
        history.append(stats['energy_consumption'])
        self.round_stats = stats
        info.update(selection_mode='independent_local', local_reward=reward)
        return arm, info, context, response

    def evaluate_local(self):
        mode = self.model.training
        self.model.to(self.device)
        self.model.eval()
        self._synchronize()
        start = time.perf_counter()
        # Use the framework's micro AUC and sample-weighted CE loss.
        # Evaluation must not change the next round's sampling randomness.
        with torch.random.fork_rng(devices=[]):
            correct, samples, auc = self.test_metrics()
            train_loss, train_samples = self.train_metrics()
        self._synchronize()
        duration = time.perf_counter()-start
        self.model.cpu()
        self.model.train(mode)
        return encode(dict(protocol=PROTOCOL, kind='evaluation', client_id=self.id,
                           correct=correct, samples=samples, trained_rounds=self.trained_rounds,
                           auc=auc, train_loss=train_loss, train_samples=train_samples,
                           evaluation_time=duration))

    def checkpoint(self, clone=True):
        # Local persistence artifact, not a federated message.
        snapshot = copy.deepcopy if clone else (lambda value: value)
        return dict(initialization_bytes=self.initialization_bytes, local_bandit=getattr(self, 'local_bandit', None), local_energy=getattr(self, 'local_energy', None), master=snapshot(self.master), model=snapshot(self.model), moments=snapshot(self.moments),
                    coordinates=copy.deepcopy(self.pruning_coordinates), retention=self.retention,
                    trained_rounds=self.trained_rounds, train_time_cost=copy.deepcopy(self.train_time_cost),
                    system_rng=copy.deepcopy(self.system_rng.bit_generator.state),
                    system_state=copy.deepcopy(self.current_system_state), round_stats=copy.deepcopy(self.round_stats),
                    bandwidth_history=list(self.bandwidth_history))

    def restore(self, state):
        self.initialization_bytes = state['initialization_bytes']
        if state.get('local_bandit') is not None:
            self.local_bandit, self.local_energy = state['local_bandit'], state['local_energy']
        self.master, self.model = state['master'], state['model']
        self.moments, self.pruning_coordinates = state['moments'], state['coordinates']
        self.retention, self.trained_rounds = state['retention'], state['trained_rounds']
        self.train_time_cost = state['train_time_cost']
        self.system_rng.bit_generator.state = state['system_rng']
        self.current_system_state, self.round_stats = state['system_state'], state['round_stats']
        self.bandwidth_history = state['bandwidth_history']
        self.optimizer = None
