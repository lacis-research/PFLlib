"""HERAFL: FedSC global training with structured pruning and prototype exchange."""
import copy
import json
import math
import random
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from flcore.servers.serverherafl import HERAFL
from flcore.servers.serverfedload import FedLoad
from flcore.clients.clientheraprototype import clientHERAPrototype
from flcore.heterogeneity.linucb_fedload import LinUCBBandit
from flcore.heterogeneity.semantic import SemanticLinUCB, history_context
from flcore.heterogeneity.pruning import prune_model, aggregate_submodels
from flcore.heterogeneity.prototype_protocol import (
    PROTOCOL, encode, decode, empty_reference, read_payload, build_reference, resource_stats, state_wire_size, FEDSC_REFERENCE,
)


class PrototypeHERAFL(HERAFL):
    client_class = clientHERAPrototype
    bandit_class = LinUCBBandit

    def __init__(self, args, times):
        from flcore.trainmodel.models import FedAvgCNN, DNN
        from flcore.trainmodel.vgg import VGG11, VGG16
        if type(args.model) not in (FedAvgCNN, DNN, VGG11, VGG16):
            raise ValueError('Prototype mode supports FedAvgCNN, DNN, VGG11 and VGG16.')
        if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in args.model.modules()):
            raise ValueError('Prototype pruning does not support BatchNorm buffers.')
        if not 0 < getattr(args, 'reward_validation_fraction', 0.) < 1:
            raise ValueError('Prototype mode requires --reward-validation-fraction in (0,1); test cannot drive the policy.')
        self.local_only = getattr(args, 'hera_local_only', False)
        if self.local_only and (getattr(args, 'hera_semantic_memory_limit', None) or getattr(args, 'hera_semantic_time_limit', None) or getattr(args, 'hera_prototype_policy', 'resources') != 'resources'):
            raise ValueError('Independent local baseline supports resource policy without optional feasibility filters.')
        self.kind = getattr(args, 'hera_prototype_kind', 'simple')
        if self.kind not in ('simple', 'relational'):
            raise ValueError('Unknown prototype aggregation.')
        self.max_age = getattr(args, 'hera_semantic_max_age', 5)
        self.warmup = getattr(args, 'hera_semantic_warmup_rounds', 1)
        self.neighbors = getattr(args, 'hera_semantic_neighbors', 1)
        if any(type(v) is not int or v < 0 for v in (self.max_age, self.warmup, self.neighbors)):
            raise ValueError('Age, warmup and neighbors must be nonnegative integers.')
        self.options = dict(alignment=getattr(args, 'hera_align_weight', 1.) if self.kind == 'simple' else 0.,
                            rpcl=getattr(args, 'hera_rpcl_weight', 0.), cpdr=getattr(args, 'hera_cpdr_weight', 0.),
                            temperature=getattr(args, 'hera_semantic_temperature', .05))
        if self.local_only:
            self.options.update(alignment=0., rpcl=0., cpdr=0.)
        if self.kind == 'simple' and (self.options['rpcl'] or self.options['cpdr']):
            raise ValueError('RPCL/CPDR require --hera-prototype-kind relational.')
        if any(not math.isfinite(v) or v < 0 for v in self.options.values()) or self.options['temperature'] <= 0:
            raise ValueError('Loss weights must be finite/nonnegative and temperature positive.')
        if getattr(args, 'hera_semantic_reward_weight', 0):
            raise ValueError('Prototype protocol uses the control validation reward; semantic reward is not supported.')
        self.semantic_features = getattr(args, 'hera_semantic_features', 'H,U,D,Q').split(',')
        if not self.semantic_features or set(self.semantic_features)-set('HUDQ'):
            raise ValueError('Invalid semantic features.')
        self.semantic_enabled = False  # distinct schema, not legacy result renderer
        self.semantic_collect = True
        self.use_semantic_policy = getattr(args, 'hera_prototype_policy', 'resources') == 'semantic'
        if self.use_semantic_policy:
            self.bandit_class = SemanticLinUCB
        self.memory_limit = getattr(args, 'hera_semantic_memory_limit', None)
        self.time_limit = getattr(args, 'hera_semantic_time_limit', None)
        for limit in (self.memory_limit, self.time_limit):
            if limit is not None and (not math.isfinite(limit) or limit <= 0):
                raise ValueError('Resource limits must be positive and finite.')
        FedLoad.__init__(self, args, times)
        self.model_name = type(self.global_model).__name__
        self.model_shapes = {k: tuple(v.shape) for k, v in self.global_model.state_dict().items()}
        self.dimension = self.global_model.fc.in_features
        template = self.global_model.cpu()
        self.client_storage = getattr(args, 'hera_client_storage', 'memory')
        if self.client_storage not in ('disk', 'memory'):
            raise ValueError('hera_client_storage must be disk or memory.')
        self.state_directory = None
        if self.client_storage == 'disk':
            import tempfile
            # Use the workspace disk, not /tmp (which may be a RAM filesystem).
            parent = Path(getattr(args, 'hera_state_dir', None) or Path(__file__).resolve().parents[3]/'tmp')
            parent.mkdir(parents=True, exist_ok=True)
            self.state_directory = tempfile.TemporaryDirectory(prefix='hera-state-', dir=parent)
        for client in self.clients:
            client.configure_storage(Path(self.state_directory.name)/f'client_{client.id}.pt'
                                     if self.state_directory else None)
        local_config = dict(arms=args.pruning_arms, alpha=args.bandit_alpha, beta=args.bandit_beta, alpha_strategy=getattr(args, 'bandit_alpha_strategy', 'binary'), seed=self.experiment_seed, max_samples=self.max_samples, num_clients=self.num_clients, reward_weights=self.reward_weights) if self.local_only else None
        resuming = bool(getattr(args, 'hera_resume', None))
        initialization = b'' if resuming else encode(dict(protocol=PROTOCOL, kind='initialization', state=template.state_dict(), local_config=local_config))
        initialization_message = decode(initialization) if not resuming else None
        self.initialization_bytes = len(initialization)*len(self.clients)
        self.initialization_client_stats = {}
        for client in self.clients:
            if not resuming:
                client.initialize_local(template, initialization, message=initialization_message)
                client.offload_state()
            else:
                client.local_config = local_config
            seconds = len(initialization)*8/(client.profile.get('sampled_throughput_mbps', client.profile.get('throughput_mbps', 10.))*1e6)
            self.initialization_client_stats[client.id] = dict(
                bytes=len(initialization), time=seconds,
                energy=seconds*client.profile.get('e_down_w', client.profile.get('e_down', .5)))
        del initialization_message, initialization
        self.costs = {}
        full = sum(p.numel()*p.element_size() for p in template.parameters())
        for arm in self.bandit.pruning_rates:
            sub, _ = prune_model(template, arm, device='cpu', preserve_representation=True)
            size = sum(p.numel()*p.element_size() for p in sub.parameters())
            self.costs[arm] = dict(bytes=size, fraction=size/full,
                                   predicted_memory=size*(5 if getattr(template, 'optimizer_kind', 'sgd') == 'adam' else 3))
            del sub
        # The independent-local ablation is the only mode without global weights.
        self.global_model = None if self.local_only else template
        if self.use_semantic_policy:
            self.bandit.legacy_policy = copy.deepcopy(self.bandit)
            self.bandit.enable(getattr(args, 'hera_semantic_policy', 'linucb'))
        self.reference = empty_reference(self.dimension)
        self.history = {}
        self.eval_rounds = []
        self.checkpoint_every = getattr(args, 'hera_checkpoint_every', 0)
        if self.checkpoint_every < 0:
            raise ValueError('Checkpoint interval must be nonnegative.')
        if getattr(args, 'hera_resume', None):
            self.load_checkpoint(Path(args.hera_resume))

    def snapshot(self):
        reference = copy.deepcopy(self.reference)
        if self.current_round <= self.warmup or self.current_round-reference['round'] > self.max_age:
            reference = empty_reference(self.dimension)
        if getattr(self.args, 'hera_semantic_shuffle', False) and reference['prototypes']:
            labels = sorted(reference['prototypes'], key=int)
            shuffled = np.random.default_rng(self.experiment_seed+self.current_round).permutation(len(labels))
            reference['prototypes'] = {k: reference['prototypes'][labels[j]] for k, j in zip(labels, shuffled)}
            if reference['relations']:
                reference['relations'] = {k: reference['relations'][labels[j]] for k, j in zip(labels, shuffled)}
        return reference

    def prepare_contexts(self):
        contexts = FedLoad.prepare_contexts(self)
        if self.use_semantic_policy:
            for i, context in contexts.items():
                context.update(history_context(self.history.get(i), self.current_round, self.max_age, self.semantic_features))
        return contexts

    @property
    def evaluation_type(self):
        return 'local_models_sample_weighted' if self.local_only else 'global_model'

    def request(self, client_id, retention, reference, include_state=True):
        return encode(dict(protocol=PROTOCOL, kind='train', client_id=client_id, round=self.current_round,
                           retention=retention, reference=reference, options=self.options,
                           state=None if self.local_only or not include_state else self.global_model.state_dict(),
                           seed=self.experiment_seed+self.current_round*self.num_clients+client_id if self.aligned_comparison else None,
                           system_seed=[self.experiment_seed, self.current_round, client_id]))

    def choose_retention(self, client_id, context, pending, reference):
        # Upper estimate of class coverage + metadata, independent of parameter bytes.
        upload_estimate = self.num_classes*(self.dimension*4*4/3+128)+4096
        costs = {}
        download_weights = 0 if self.local_only else state_wire_size(self.global_model.state_dict())-4
        for arm, spec in self.costs.items():
            previous = next(c.round_stats for c in self.clients if c.id == client_id)
            compute = (previous['compute_time']*spec['fraction']/max(previous.get('parameter_fraction', 1), 1e-12)
                       if previous else self.max_samples*.002*(self.local_epochs+1)*spec['fraction'])
            # JSON float32/base64 transport, including global download and submodel upload.
            parameter_bytes = 0 if self.local_only else spec['bytes']*4/3
            seconds = compute+(len(self.request(client_id, arm, reference, include_state=False))+download_weights+upload_estimate+parameter_bytes)*8/(context['throughput_mbps']*1e6)
            costs[arm] = dict(spec, predicted_time=seconds, feasible=(self.memory_limit is None or spec['predicted_memory'] <= self.memory_limit)
                              and (self.time_limit is None or seconds <= self.time_limit))
        if self.use_semantic_policy:
            return self.bandit.choose_semantic(context, pending, costs,
                                              self.current_round <= self.warmup or not context.get('semantic_available', 0))
        feasible = [a for a in self.bandit.pruning_rates if costs[a]['feasible']]
        if not feasible:
            return None, dict(selection_mode='no_feasible_arm', candidate_costs=costs)
        # Same resource policy, restricted to physically feasible actions.
        original = self.bandit.pruning_rates
        try:
            self.bandit.pruning_rates = feasible
            arm, info = self.bandit.choose_pruning_rate(context, pending)
        finally:
            self.bandit.pruning_rates = original
        return arm, dict(info, candidate_costs=costs)

    def evaluate(self, *unused, **kwargs):
        metrics, seconds, byte_count, energy, transfer = [], 0., 0, 0., 0.
        parameter_bytes = 0 if self.local_only else state_wire_size(self.global_model.state_dict())
        for client in self.clients:
            with client.resident(write=False):
                local_model = client.model
                try:
                    if not self.local_only:
                        client.model = self.global_model
                    message = client.evaluate_local(detailed=not self.local_only and getattr(self.args, 'detailed_evaluation', False))
                finally:
                    client.model = local_model
            record = decode(message)
            byte_count += len(message)
            seconds += record['evaluation_time']
            transmission = len(message)*8/(client.profile.get('sampled_throughput_mbps', client.profile.get('throughput_mbps', 10.))*1e6)
            download = parameter_bytes*8/(client.profile.get('sampled_throughput_mbps', client.profile.get('throughput_mbps', 10.))*1e6)
            transfer += transmission+download
            energy += download*client.profile.get('e_down_w', client.profile.get('e_down', .5))
            energy += transmission*client.profile.get('e_up_w', client.profile.get('e_up', .5)) + record['evaluation_time']*client.profile.get('e_cmp_w', client.profile.get('e_cmp', .5))
            record['accuracy'] = record['correct']/record['samples'] if record['samples'] else None
            metrics.append(record)
        valid = [r['accuracy'] for r in metrics if r['accuracy'] is not None]
        total = sum(r['samples'] for r in metrics)
        weighted = sum(r['correct'] for r in metrics)/max(total, 1)
        auc = sum(r['auc']*r['samples'] for r in metrics)/max(total, 1)
        train_loss = sum(r['train_loss'] for r in metrics)/max(sum(r['train_samples'] for r in metrics), 1)
        auc_std = float(np.std([r['auc'] for r in metrics]))
        print(f'Averaged Train Loss: {train_loss:.4f}')
        print(f'Averaged Test Accuracy: {weighted:.4f}')
        print(f'Averaged Test AUC: {auc:.4f}')
        print(f'Std Test Accuracy: {np.std(valid):.4f}')
        print(f'Std Test AUC: {auc_std:.4f}')
        self.rs_test_acc.append(weighted)
        self.rs_train_loss.append(train_loss)
        self.rs_test_auc.append(auc)
        self.eval_rounds.append(self.current_round)
        details = {}
        if metrics and all('confusion' in r for r in metrics):
            from flcore.heterogeneity.semantic import summarize_confusion
            total = sum(torch.tensor(r['confusion'], dtype=torch.int64) for r in metrics)
            details['test_sample_weighted'] = summarize_confusion(total)
            for record in metrics:
                record.update(summarize_confusion(torch.tensor(record['confusion'], dtype=torch.int64)))
        return dict(evaluation_type=self.evaluation_type,
                    **({'local_accuracy': weighted} if self.local_only else {'global_accuracy': weighted}),
                    train_loss=train_loss, test_auc=auc, test_auc_std=auc_std,
                    test_clients=metrics, test_client_mean=float(np.mean(valid)) if valid else None,
                    test_client_median=float(np.median(valid)) if valid else None,
                    test_client_p10=float(np.percentile(valid, 10)) if valid else None,
                    test_client_std=float(np.std(valid)) if valid else None,
                    evaluation_clients=len(metrics), evaluation_time=seconds, evaluation_telemetry_bytes=byte_count,
                    evaluation_parameter_bytes=parameter_bytes*len(metrics),
                    evaluation_communication_bytes=byte_count+parameter_bytes*len(metrics),
                    evaluation_total_time=seconds+transfer, evaluation_energy=energy, **details)

    def train(self):
        for index in range(self.current_round+1, self.global_rounds+1):
            self.current_round = index
            print(f'\n-------------Round number: {index}-------------', flush=True)
            start = time.perf_counter()
            self.selected_clients = self.select_clients()
            contexts, reference = self.prepare_contexts(), self.snapshot()
            pending, assignments, local_traces = [], {}, {}
            for client in self.selected_clients:
                if self.local_only:
                    with client.resident():
                        arm, info, contexts[client.id], trace = client.local_round(index)
                    local_traces[client.id] = trace
                else:
                    arm, info = self.choose_retention(client.id, contexts[client.id], pending, reference)
                assignments[client.id] = (arm, info)
                if arm is not None:
                    pending.append(arm)
            active_count = int((1-self.client_drop_rate)*len(self.selected_clients))
            active = {c.id for c in random.Random(self.experiment_seed+index).sample(self.selected_clients, active_count)}
            reports, accepted_payloads, accepted_updates = [], [], []
            for client in self.selected_clients:
                arm, info = assignments[client.id]
                report = dict(id=client.id, retention=arm, selection_context=contexts[client.id], alpha_info=info,
                              num_samples=client.train_samples, status='infeasible', accuracy=None, round_stats={})
                reports.append(report)
                if arm is None:
                    continue
                request = self.request(client.id, arm, reference)
                if self.local_only:
                    response = local_traces[client.id]
                else:
                    with client.resident():
                        response = client.exchange(request)
                if index % self.eval_gap == 0:
                    with client.resident(write=False):
                        client.model.to(client.device)
                        report['local_test'] = client.local_test_accuracy_metrics()
                        client.model.cpu()
                message = decode(response)
                if set(message) != {'protocol', 'kind', 'payload', 'telemetry', 'state'} or message['protocol'] != PROTOCOL or message['kind'] != 'response':
                    raise ValueError('Unexpected response type.')
                payload = read_payload(message['payload'], self.dimension, self.num_classes, index, client.id)
                telemetry = message['telemetry']
                report['loss_diagnostics'] = telemetry.get('loss_diagnostics', [])
                stats = resource_stats(telemetry, 0 if self.local_only else len(response), 0 if self.local_only else len(request), client.profile)
                # Exact wire decomposition: semantic object bytes + envelope/control remainder.
                stats['semantic_download_bytes'] = len(encode(reference))
                stats['semantic_upload_bytes'] = len(encode(message['payload']))
                stats['parameter_download_bytes'] = 0 if self.local_only else state_wire_size(self.global_model.state_dict())
                stats['parameter_upload_bytes'] = 0 if self.local_only else state_wire_size(message['state'])
                stats['control_download_bytes'] = len(request)-stats['semantic_download_bytes']-stats['parameter_download_bytes']
                stats['control_upload_bytes'] = len(response)-stats['semantic_upload_bytes']-stats['parameter_upload_bytes']
                if self.local_only:
                    for key in ('semantic_download_bytes', 'semantic_upload_bytes', 'control_download_bytes', 'control_upload_bytes'):
                        stats[key] = 0
                client.round_stats = stats  # resource telemetry, not model access
                client.current_system_state = telemetry['system_state']
                self.contexts[client.id] = client.get_context(self.max_samples)
                accuracy = telemetry['validation_correct']/max(telemetry['validation_samples'], 1)
                timeout = min(self.base_round_timeout, self.time_threthold if self.time_select else math.inf)
                accepted = self.local_only or (client.id in active and stats['round_time'] <= timeout)
                report.update(status='accepted' if accepted else ('dropped' if client.id not in active else 'timeout'),
                              accuracy=accuracy, accuracy_source='held_out_train', context=dict(self.contexts[client.id]),
                              round_stats=stats, semantic_payload=payload.metadata())
                if accepted and not self.local_only:
                    accepted_payloads.append(payload)
                    # Reconstruct from the same global weights/retention, not client cache.
                    submodel, coordinates = prune_model(self.global_model, arm, device='cpu', preserve_representation=True)
                    submodel.load_state_dict(message['state'], strict=True)
                    accepted_updates.append((submodel, coordinates, client.train_samples))
            accepted = [r for r in reports if r['status'] == 'accepted']
            time_reference = float(np.mean([r['round_stats']['round_time'] for r in accepted])) if accepted else 0.
            for report in ([] if self.local_only else accepted):
                arm = report['retention']
                reward, components = FedLoad.calculate_reward(self, arm, report['accuracy'], report['round_stats'], time_reference)
                report.update(reward=reward, reward_components=components, learning_context=report['selection_context'])
                self.bandit.update(arm, reward, report['selection_context'])
                if self.use_semantic_policy:
                    self.bandit.legacy_policy.update(arm, reward, report['selection_context'])
            for report in accepted:
                self.arm_energy_history[report['retention']].append(report['round_stats']['energy_consumption'])
            aggregate_start = time.perf_counter()
            if not self.local_only:
                self.global_model = aggregate_submodels(self.global_model, accepted_updates)
            parameter_aggregation_time = time.perf_counter()-aggregate_start
            aggregate_start = time.perf_counter()
            self.reference = build_reference(accepted_payloads, self.reference, index, self.num_classes, self.kind, self.neighbors, device=self.device)
            aggregation_time = time.perf_counter()-aggregate_start
            self.history.update({p.client_id: p for p in accepted_payloads})
            record = dict(round=index, clients=reports, protocol=PROTOCOL, global_accuracy=None,
                          parameter_aggregation_time=parameter_aggregation_time,
                          semantic_aggregation_time=aggregation_time, semantic_bank_round=self.reference['round'],
                          round_timeout=None if math.isinf(self.base_round_timeout) else self.base_round_timeout)
            if index == 1:
                record.update(initialization_bytes=self.initialization_bytes,
                              initialization_time=sum(v['time'] for v in self.initialization_client_stats.values()),
                              initialization_energy=sum(v['energy'] for v in self.initialization_client_stats.values()))
            if index % self.eval_gap == 0:
                record.update(self.evaluate())
            record['round_wall_time'] = time.perf_counter()-start
            self.round_reports.append(record)
            self.Budget.append(record['round_wall_time'])
            print('Retentions:', {r['id']: r['retention'] for r in reports})
            print('Accepted updates:', len(accepted), '| time cost:', self.Budget[-1], flush=True)
            if self.checkpoint_every and index % self.checkpoint_every == 0:
                self.save_checkpoint(self.output_directory()/'checkpoint')
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        # Only discard training state after every result/plot was saved.
        self.save_results()
        if getattr(self.args, 'hera_keep_checkpoints', False):
            self.save_checkpoint(self.output_directory()/'checkpoint')
        self.cleanup_training_artifacts()

    def cleanup_training_artifacts(self):
        """Remove only this run's cache/checkpoints after successful export."""
        import shutil
        if not getattr(self.args, 'hera_keep_checkpoints', False):
            checkpoint = self.output_directory()/'checkpoint'
            if checkpoint.exists():
                shutil.rmtree(checkpoint)
        if self.state_directory is not None:
            self.state_directory.cleanup()
            for client in self.clients:
                client.release_state()
            self.state_directory = None
        print('Training cache removed; ' + (
            'checkpoints preserved.' if getattr(self.args, 'hera_keep_checkpoints', False)
            else 'checkpoints removed. Metrics and plots preserved.'), flush=True)

    def config_signature(self):
        ignored = {'model', 'global_rounds', 'goal', 'experiment_id', 'hera_resume', 'hera_checkpoint_every', 'device', 'device_id', 'hera_client_storage', 'hera_state_dir', 'hera_keep_checkpoints'}
        return {k: v for k, v in vars(self.args).items() if k not in ignored and isinstance(v, (str, int, float, bool, list, type(None)))}

    def output_directory(self):
        if not hasattr(self, 'result_directory'):
            from datetime import datetime
            from zoneinfo import ZoneInfo
            import re
            stamp = datetime.now(ZoneInfo('America/Belem')).strftime('%Y%m%d_%Hh%Mm%Ss_%f')
            experiment = re.sub(r'[^a-zA-Z0-9_.-]', '_', getattr(self.args, 'experiment_id', self.goal))
            self.result_directory = Path('../results/herafl')/f'{stamp}_prototypes_{self.kind}_{experiment}_run{self.times}'
        self.result_directory.mkdir(parents=True, exist_ok=True)
        return self.result_directory

    def save_checkpoint(self, path):
        path.mkdir(parents=True, exist_ok=True)
        root = path
        import tempfile
        path = Path(tempfile.mkdtemp(prefix=f'round_{self.current_round:06d}_', dir=root))
        import errno
        import os
        import shutil
        try:
            # Local snapshots are not learning messages. The cache uses atomic
            # replacement, so hard links remain immutable when training continues.
            for client in self.clients:
                if getattr(client, 'state_path', None) is not None:
                    tensor_path = path/f'client_{client.id}_tensors.pt'
                    try:
                        os.link(client.state_path, tensor_path)
                    except OSError as error:
                        if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS):
                            raise
                        shutil.copyfile(client.state_path, tensor_path)
                    metadata = client.checkpoint(clone=False)
                    for key in ('master', 'model', 'moments', 'coordinates'):
                        metadata.pop(key)
                    metadata['tensor_file'] = tensor_path.name
                    torch.save(metadata, path/f'client_{client.id}.pt')
                else:
                    torch.save(client.checkpoint(clone=False), path/f'client_{client.id}.pt')
            state = dict(protocol=PROTOCOL, signature=self.config_signature(), model=self.model_name, model_shapes=self.model_shapes, round=self.current_round,
                         global_state=None if self.local_only else self.global_model.state_dict(),
                         bandit=self.bandit, reference=self.reference, history=self.history, contexts=self.contexts,
                         energy=self.arm_energy_history, reports=self.round_reports, accuracies=self.rs_test_acc,
                         eval_rounds=self.eval_rounds, budget=self.Budget, initialization_bytes=self.initialization_bytes,
                         initialization_stats=self.initialization_client_stats, rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                         numpy_rng=np.random.get_state(), python_rng=random.getstate())
            torch.save(state, path/'server.tmp')
            (path/'server.tmp').replace(path/'server.pt')
            (root/'latest.tmp').write_text(json.dumps({'directory': path.name}))
            (root/'latest.tmp').replace(root/'latest.json')
        except Exception as error:
            # Remove only this unpublished generation; preserve prior snapshots.
            shutil.rmtree(path, ignore_errors=True)
            (root/'latest.tmp').unlink(missing_ok=True)
            if isinstance(error, (OSError, RuntimeError)):
                free_gib = shutil.disk_usage(root).free / 2**30
                raise RuntimeError(
                    f'Checkpoint could not be saved in {root} ({free_gib:.2f} GiB free). '
                    'Check disk space, quota and filesystem permissions. '
                    'The incomplete generation was removed; prior checkpoints and '
                    'any saved metrics/plots were preserved.'
                ) from error
            raise

    def load_checkpoint(self, path):
        if (path/'latest.json').exists():
            path = path / json.loads((path/'latest.json').read_text())['directory']
        state = torch.load(path/'server.pt', map_location='cpu', weights_only=False)
        if state['protocol'] != PROTOCOL or state['signature'] != self.config_signature() or state['model'] != self.model_name or state['model_shapes'] != self.model_shapes:
            raise ValueError('Incompatible prototype checkpoint configuration.')
        if not self.local_only:
            self.global_model.load_state_dict(state['global_state'], strict=True)
        for client in self.clients:
            client_state = torch.load(path/f'client_{client.id}.pt', map_location='cpu', weights_only=False)
            if 'tensor_file' in client_state:
                tensor_file = client_state.pop('tensor_file')
                if tensor_file != f'client_{client.id}_tensors.pt':
                    raise ValueError('Invalid checkpoint tensor filename.')
                client_state.update(torch.load(path/tensor_file, map_location='cpu', weights_only=False))
            client.restore(client_state)
            del client_state
            client.offload_state()
        self.current_round, self.bandit = state['round'], state['bandit']
        if self.global_rounds < self.current_round:
            raise ValueError('Requested rounds precede checkpoint.')
        self.reference, self.history, self.contexts = state['reference'], state['history'], state['contexts']
        self.arm_energy_history, self.round_reports = state['energy'], state['reports']
        self.rs_test_acc, self.eval_rounds, self.Budget = state['accuracies'], state['eval_rounds'], state['budget']
        self.rs_train_loss = [r['train_loss'] for r in self.round_reports if r['round'] in self.eval_rounds]
        self.rs_test_auc = [r['test_auc'] for r in self.round_reports if r['round'] in self.eval_rounds]
        self.initialization_bytes, self.initialization_client_stats = state['initialization_bytes'], state['initialization_stats']
        torch.set_rng_state(state['rng'])
        if state['cuda_rng'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])

    def save_results(self):
        directory = self.output_directory()
        stem = f'{self.dataset}_{self.algorithm}_{self.goal}_{self.times}'
        config = dict(protocol=PROTOCOL, evaluation_type=self.evaluation_type, model=self.model_name,
                      training_config=self.config_signature(), initialization='one_time_weight_distribution' if self.local_only else 'global_weights_each_round',
                      loss_definition='fedsc-official', fedsc_reference=FEDSC_REFERENCE,
                      initialization_bytes=self.initialization_bytes, prototype_options=self.options,
                      prototype_kind=self.kind, local_only=self.local_only, resume_supported=True,
                      local_storage=self.client_storage,
                      local_storage_details='CPU master and last submodel; optimizer reset on global download; one resident client in disk mode',
                      initialization_client_stats=self.initialization_client_stats)
        partition = Path('../dataset')/self.dataset/'config.json'
        config['partition_metadata'] = json.loads(partition.read_text()) if partition.exists() else None
        (directory/'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
        (directory/f'{stem}_policy.json').write_text(json.dumps(self.round_reports, indent=2, allow_nan=False))
        with h5py.File(directory/f'{stem}.h5', 'w') as stream:
            stream.attrs['evaluation_type'] = self.evaluation_type
            stream.attrs['protocol'] = PROTOCOL
            stream.attrs['variant'] = f'{PROTOCOL}:loss=fedsc-official-{FEDSC_REFERENCE}:{self.kind}:local_only={self.local_only}:semantic_policy={self.use_semantic_policy}'
            stream.create_dataset('rs_local_test_acc' if self.local_only else 'rs_test_acc', data=self.rs_test_acc)
            stream.create_dataset('eval_rounds', data=self.eval_rounds)
            evaluations = {r['round']: r for r in self.round_reports}
            for name, key in (('rs_train_loss', 'train_loss'), ('rs_test_auc', 'test_auc'),
                              ('rs_test_acc_std', 'test_client_std'), ('rs_test_auc_std', 'test_auc_std')):
                if all(key in evaluations[r] for r in self.eval_rounds):
                    stream.create_dataset(name, data=[evaluations[r][key] for r in self.eval_rounds])
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'tools'))
        from comparison_metrics import render_reference_comparison
        render_reference_comparison(
            {self.algorithm: [dict(zip(self.eval_rounds, self.rs_test_acc))]},
            {self.algorithm: [(self.times, self.round_reports)]}, directory/'plots', ['png'],
            evaluation_types={self.algorithm: self.evaluation_type})
        if not self.local_only:
            from plot_comparison import render_global_metrics
            metrics = {}
            for name, key, scale in [('rs_test_acc', 'global_accuracy', 100.),
                                     ('rs_train_loss', 'train_loss', 1.), ('rs_test_auc', 'test_auc', 1.)]:
                series = {r['round']: r[key]*scale for r in self.round_reports if r.get(key) is not None}
                metrics[name] = {self.algorithm: [series]}
            render_global_metrics(metrics, {self.algorithm: self.evaluation_type}, directory/'plots')
            torch.save(self.global_model.state_dict(), directory/'global_model.pt')
