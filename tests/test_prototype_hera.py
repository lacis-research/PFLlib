"""FedSC global cycle, prototype losses, pruning and checkpoint contracts."""
import copy
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'system'))
import test_heterogeneous_algorithms as native_tests
from flcore.servers.serverherafl import HERAFL
from flcore.servers.serverheraprototype import PrototypeHERAFL
from flcore.heterogeneity.prototype_protocol import (
    encode, decode, empty_reference, build_reference, semantic_losses, read_payload,
)
from flcore.heterogeneity.semantic import Payload


class PrototypeTests(unittest.TestCase):
    make_args = native_tests.NativeServerTests.make_args

    def args(self, **options):
        defaults = dict(algorithm='HERAFL', hera_communication='prototypes', hera_client_storage='memory', hera_keep_checkpoints=True,
                        reward_validation_fraction=.2, aligned_comparison=True,
                        wait_for_all=True, pruning_arms=[.5, 1.], global_rounds=3)
        defaults.update(options)
        return self.make_args(**defaults)

    @contextlib.contextmanager
    def environment(self):
        torch.manual_seed(10)
        data = [(torch.randn(4), torch.tensor(i % 3)) for i in range(18)]
        with tempfile.TemporaryDirectory() as tmp, \
             patch('flcore.servers.serverbase.read_client_data', return_value=data), \
             patch('flcore.clients.clientbase.read_client_data', return_value=data):
            yield Path(tmp)

    def run_training(self, args, tmp):
        server = HERAFL(args, 0)
        self.assertIsInstance(server, PrototypeHERAFL)
        server.result_directory = tmp
        with patch.object(server, 'save_results'), patch.object(server, 'cleanup_training_artifacts'):
            server.train()
        return server

    def test_global_weighted_aggregation_uses_wire_updates_and_rebroadcasts(self):
        from flcore.clients.clientheraprototype import clientHERAPrototype
        original_exchange = clientHERAPrototype.exchange
        with self.environment() as tmp:
            server = HERAFL(self.args(global_rounds=2, join_ratio=1., pruning_arms=[1.], detailed_evaluation=True), 0)
            server.result_directory = tmp
            initial = copy.deepcopy(server.global_model.state_dict())
            responses, requests = {}, {}
            for client in server.clients:
                client.train_samples = client.id+1
            def exchange(client, request):
                message = decode(request)
                requests[(message['round'], client.id)] = message['state']
                reply = decode(original_exchange(client, request))
                # Deliberately differ from the local cache: aggregation must use the wire.
                reply['state'] = {k: v+client.id*.1 for k, v in reply['state'].items()}
                responses[(message['round'], client.id)] = reply['state']
                return encode(reply)
            with patch.object(clientHERAPrototype, 'exchange', exchange), patch.object(server, 'save_results'):
                server.train()
            for name in initial:
                for client in server.clients:
                    torch.testing.assert_close(requests[(1, client.id)][name], initial[name])
                round_one = sum(responses[(1, c.id)][name]*c.train_samples for c in server.clients)/6
                for client in server.clients:
                    torch.testing.assert_close(requests[(2, client.id)][name], round_one)
                expected = sum(responses[(2, c.id)][name]*c.train_samples for c in server.clients)/6
                torch.testing.assert_close(server.global_model.state_dict()[name], expected)
            before = copy.deepcopy(server.global_model.state_dict())
            record = server.evaluate()
            self.assertEqual(record['evaluation_type'], 'global_model')
            self.assertAlmostEqual(record['test_sample_weighted']['accuracy'], record['global_accuracy'])
            self.assertEqual(record['evaluation_communication_bytes'], record['evaluation_telemetry_bytes']+record['evaluation_parameter_bytes'])
            for name in before:
                torch.testing.assert_close(server.global_model.state_dict()[name], before[name])

    def test_rejected_updates_preserve_global_and_checkpoint_restores_it(self):
        with self.environment() as tmp:
            args = self.args(wait_for_all=False, pruning_round_timeout=1e-12, global_rounds=1)
            server = self.run_training(args, tmp)
            for name, value in args.model.state_dict().items():
                torch.testing.assert_close(server.global_model.state_dict()[name], value)
            args.hera_resume = str(tmp/'checkpoint')
            resumed = HERAFL(args, 0)
            for name, value in server.global_model.state_dict().items():
                torch.testing.assert_close(resumed.global_model.state_dict()[name], value)
            self.assertEqual(resumed.rs_train_loss, server.rs_train_loss)
            self.assertEqual(resumed.rs_test_auc, server.rs_test_auc)
            root = tmp/'checkpoint'
            generation = root/json.loads((root/'latest.json').read_text())['directory']
            state = torch.load(generation/'server.pt', weights_only=False)
            state['protocol'] = 'herafl-prototypes-v1'
            torch.save(state, generation/'server.pt')
            with self.assertRaisesRegex(ValueError, 'Incompatible prototype checkpoint'):
                HERAFL(args, 0)

    def test_official_cpdr_uses_raw_feature_mse(self):
        features = torch.tensor([[3., 4.], [20., 30.]], requires_grad=True)
        cpdr, _ = semantic_losses(features, torch.tensor([0, 2]),
                                  dict(empty_reference(2), prototypes={0: torch.tensor([1., 1.])}))
        self.assertEqual(float(cpdr), 3.25)
        cpdr.backward()
        torch.testing.assert_close(features.grad[1], torch.zeros(2))

    def test_state_wire_size_is_exact(self):
        from flcore.heterogeneity.prototype_protocol import state_wire_size
        for count in (1, 2, 3, 17):
            state = {'weights': torch.randn(count, 3), 'bias': torch.randn(count)}
            self.assertEqual(state_wire_size(state), len(encode(state)))

    def test_cpdr_only_ablation_does_not_require_negative_pool(self):
        reference = dict(empty_reference(2), prototypes={0: torch.ones(2)}, relations={0: torch.ones(1, 2)})
        cpdr, rpcl = semantic_losses(torch.zeros(1, 2), torch.tensor([0]), reference, compute_contrastive=False)
        self.assertEqual(float(cpdr), 1.)
        self.assertEqual(float(rpcl), 0.)

    def test_default_storage_trains_without_disk_state_io(self):
        with self.environment() as tmp:
            args = self.args(global_rounds=2, join_ratio=1., hera_keep_checkpoints=False)
            del args.hera_client_storage
            with patch('torch.save', side_effect=AssertionError('state write')), \
                 patch('torch.load', side_effect=AssertionError('state read')):
                server = self.run_training(args, tmp)
            self.assertEqual(server.client_storage, 'memory')
            self.assertIsNone(server.state_directory)
            self.assertTrue(server.reference['prototypes'])
            self.assertEqual(server.eval_rounds, [1, 2])
            for client in server.clients:
                self.assertIsNone(client.state_path)
                self.assertIsNotNone(client.model)
                self.assertEqual(client.master.fc.weight.device.type, 'cpu')
                self.assertEqual(client.trained_rounds, 2)

    def test_protocol_persistence_pruning_and_local_evaluation(self):
        with self.environment() as tmp:
            args = self.args(join_ratio=1.)
            # Forced reduce/increase exercises both directions without a global reset.
            def choice(server, cid, context, pending, ref):
                return ([.5, 1., .5][server.current_round-1], {'selection_mode': 'test'})
            with patch.object(PrototypeHERAFL, 'choose_retention', choice):
                server = self.run_training(args, tmp)
            self.assertIsNotNone(server.global_model)
            self.assertEqual(server.reference['round'], 3)
            self.assertEqual(server.eval_rounds, [1, 2, 3])
            self.assertEqual(server.round_reports[-1]['evaluation_clients'], 3)
            self.assertTrue(all(c.model is not c.master for c in server.clients))
            for r in server.round_reports:
                self.assertIsNotNone(r['global_accuracy'])
                for c in r['clients']:
                    stats = c['round_stats']
                    self.assertGreater(stats['parameter_upload_bytes'], 0)
                    self.assertGreater(stats['parameter_download_bytes'], 0)
                    self.assertEqual(stats['total_communication_bytes'], sum(stats[k] for k in ['semantic_upload_bytes', 'semantic_download_bytes', 'control_upload_bytes', 'control_download_bytes', 'parameter_upload_bytes', 'parameter_download_bytes']))
                    self.assertEqual(c['learning_context'], c['selection_context'])
            before = [copy.deepcopy(c.model.state_dict()) for c in server.clients]
            server.evaluate()
            for c, old in zip(server.clients, before):
                for name, value in c.model.state_dict().items():
                    torch.testing.assert_close(value, old[name])
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(server.clients[0].master.parameters(), args.model.parameters())))

    def test_rejected_updates_remain_local_but_not_in_bank(self):
        with self.environment() as tmp:
            args = self.args(wait_for_all=False, pruning_round_timeout=1e-12, global_rounds=2, join_ratio=1.)
            server = self.run_training(args, tmp)
            self.assertFalse(server.reference['prototypes'])
            self.assertFalse(server.history)
            self.assertTrue(all(c.trained_rounds == 2 for c in server.clients))
            self.assertTrue(all(r['round_stats']['total_communication_bytes'] > 0 for rr in server.round_reports for r in rr['clients']))
            self.assertTrue(all(r['status'] == 'timeout' for rr in server.round_reports for r in rr['clients']))

    def test_checkpoint_resumes_models_moments_and_rounds(self):
        with self.environment() as tmp:
            args = self.args(global_rounds=1, join_ratio=1.)
            args.model.optimizer_kind = 'adam'
            first = self.run_training(args, tmp/'first')
            resumed_args = copy.deepcopy(args)
            resumed_args.global_rounds = 2
            resumed_args.hera_resume = str(tmp/'first'/'checkpoint')
            second = HERAFL(resumed_args, 0)
            for old, new in zip(first.clients, second.clients):
                for name, p in old.master.named_parameters():
                    torch.testing.assert_close(p, new.master.get_parameter(name))
                self.assertTrue(new.moments)
            second.result_directory = tmp/'second'
            with patch.object(second, 'save_results'):
                second.train()
            self.assertEqual(second.eval_rounds, [1, 2])
            self.assertEqual(len(second.round_reports), 2)
            self.assertTrue(all(c.trained_rounds == 2 for c in second.clients))

    def test_relational_missing_classes_noncontiguous_ids_and_loss_gradient(self):
        p1 = Payload(4, 1, .5, 2, {0: 2, 1: 1}, {0: torch.tensor([1., 0.]), 1: torch.tensor([0., 1.])},
                     dict.fromkeys('HUDQ', .5), dict.fromkeys('HUDQ', True), 1., 0)
        p2 = Payload(19, 1, 1., 2, {0: 1}, {0: torch.tensor([.8, .2])},
                     dict.fromkeys('HUDQ', .5), dict.fromkeys('HUDQ', True), 1., 0)
        bank = build_reference([p1, p2], empty_reference(2), 1, 3, 'relational', 10)
        self.assertEqual(bank['relations'][0].shape, (2, 2))
        torch.testing.assert_close(p1.prototypes[0], torch.tensor([1., 0.]))
        self.assertNotIn(2, bank['prototypes'])
        z = torch.tensor([[.5, .8], [.8, .5]], requires_grad=True)
        align, contrast = semantic_losses(z, torch.tensor([0, 1]), bank)
        (align+contrast).backward()
        self.assertGreater(float(z.grad.abs().sum()), 0)
        self.assertFalse(bank['prototypes'][0].requires_grad)
        self.assertEqual(float(semantic_losses(z, torch.tensor([0, 1]), empty_reference(2))[0]), 0)
        decoded = decode(encode(vars(p1)))
        recovered = read_payload(decoded, 2, 3, 1, 4)
        torch.testing.assert_close(recovered.prototypes[0], p1.prototypes[0])
        with self.assertRaises(ValueError):
            read_payload(decoded, 3, 3, 1, 4)

    def test_wire_serialization_and_expiration(self):
        with self.environment() as tmp:
            server = HERAFL(self.args(hera_semantic_max_age=0), 0)
            server.current_round = 2
            server.reference = dict(round=1, dimension=8, prototypes={0: torch.ones(8)}, relations={}, contributors={0: [4]})
            self.assertFalse(server.snapshot()['prototypes'])
            request = decode(server.request(0, .5, server.snapshot()))
            self.assertEqual(set(request), {'protocol', 'kind', 'client_id', 'round', 'retention', 'reference', 'options', 'seed', 'system_seed', 'state'})
            self.assertIn('state', request)
            self.assertEqual(set(request['state']), set(server.global_model.state_dict()))

    def test_independent_local_control_has_no_learning_network(self):
        with self.environment() as tmp:
            server = self.run_training(self.args(hera_local_only=True, global_rounds=2), tmp)
            self.assertFalse(server.reference['prototypes'])
            self.assertFalse(server.history)
            self.assertTrue(all(count == 0 for count in server.bandit.arm_counts.values()))
            self.assertTrue(any(sum(c.local_bandit.arm_counts.values()) > 0 for c in server.clients))
            for rr in server.round_reports:
                self.assertTrue(all(r['round_stats']['total_communication_bytes'] == 0 for r in rr['clients']))

    def test_relational_semantic_policy_and_zero_memory_feasibility(self):
        with self.environment() as tmp:
            server = self.run_training(self.args(hera_prototype_kind='relational', hera_rpcl_weight=1.,
                                      hera_cpdr_weight=1., hera_prototype_policy='semantic', global_rounds=2), tmp/'relational')
            self.assertTrue(server.reference['relations'])
            self.assertGreater(sum(server.bandit.arm_counts.values()), 0)
            infeasible = self.run_training(self.args(hera_semantic_memory_limit=1., global_rounds=1), tmp/'infeasible')
            self.assertTrue(all(c.trained_rounds == 0 for c in infeasible.clients))
            self.assertTrue(all(r['status'] == 'infeasible' for r in infeasible.round_reports[0]['clients']))

    def test_vgg_prototype_training_and_adam_state(self):
        import gc
        from flcore.trainmodel.vgg import VGG11, VGG16
        for model_type in (VGG11, VGG16):
            data = [(torch.randn(3, 32, 32), torch.tensor(i % 3)) for i in range(6)]
            with patch('flcore.servers.serverbase.read_client_data', return_value=data), \
                 patch('flcore.clients.clientbase.read_client_data', return_value=data):
                server = HERAFL(self.args(model=model_type(num_classes=3), num_clients=1, join_ratio=1.,
                                         pruning_arms=[.0625, .125], batch_size=2, global_rounds=2,
                                         local_learning_rate=.0001, hera_prototype_kind='relational', hera_rpcl_weight=1., hera_cpdr_weight=1.), 0)
                with patch.object(server, 'save_results'), patch.object(server, 'save_checkpoint'), patch.object(server, 'output_directory', return_value=Path('/tmp/unused-prototype-test')):
                    server.train()
                self.assertTrue(server.clients[0].moments)
                self.assertEqual(server.reference['dimension'], 4096)
                self.assertTrue(server.reference['prototypes'])
                self.assertGreater(server.round_reports[-1]['clients'][0]['round_stats']['parameter_upload_bytes'], 0)
                del server
                gc.collect()

    def test_nonparticipants_persist_and_local_scatter_preserves_dormant_weights(self):
        from flcore.clients.clientheraprototype import indices
        with self.environment() as tmp:
            server = HERAFL(self.args(global_rounds=2), 0)
            client = server.clients[0]
            original = copy.deepcopy(client.master.state_dict())
            client.install_local(.5)
            coords = copy.deepcopy(client.pruning_coordinates)
            with torch.no_grad():
                for p in client.model.parameters():
                    p.add_(1.)
            client.commit_local()
            for name, value in client.master.named_parameters():
                expected = original[name].clone()
                expected[indices(coords[name])] += 1
                torch.testing.assert_close(value, expected)
            client.install_local(1.)
            for name, p in client.model.named_parameters():
                torch.testing.assert_close(p, client.master.get_parameter(name))
            client.commit_local()
            absent_before = copy.deepcopy(server.clients[2].master.state_dict())
            schedule = iter([[server.clients[0]], [server.clients[1]]])
            server.result_directory = tmp
            with patch.object(server, 'select_clients', side_effect=lambda: next(schedule)), patch.object(server, 'save_results'):
                server.train()
            self.assertEqual([c.trained_rounds for c in server.clients], [1, 1, 0])
            for name, p in server.clients[2].master.named_parameters():
                torch.testing.assert_close(p, absent_before[name])

    def test_disk_storage_matches_memory_and_resumes(self):
        def choice(server, cid, context, pending, ref):
            return ([.5, 1., .5][server.current_round-1], {'selection_mode': 'test'})

        def assert_states(left, right):
            for name, tensor in left.global_model.state_dict().items():
                torch.testing.assert_close(tensor, right.global_model.state_dict()[name], rtol=0, atol=0)
            for a, b in zip(left.clients, right.clients):
                with a.resident(write=False), b.resident(write=False):
                    for attr in ('master', 'model'):
                        for name, tensor in getattr(a, attr).state_dict().items():
                            torch.testing.assert_close(tensor, getattr(b, attr).state_dict()[name], rtol=0, atol=0)
                    self.assertEqual(a.moments.keys(), b.moments.keys())
                    for name, state in a.moments.items():
                        for key, tensor in state.items():
                            torch.testing.assert_close(tensor, b.moments[name][key], rtol=0, atol=0)
                    self.assertEqual(a.trained_rounds, b.trained_rounds)

        with self.environment() as tmp, patch.object(PrototypeHERAFL, 'choose_retention', choice):
            args = self.args(global_rounds=3, join_ratio=1.)
            args.model.optimizer_kind = 'adam'
            memory = self.run_training(copy.deepcopy(args), tmp/'memory')
            disk_args = copy.deepcopy(args)
            disk_args.hera_client_storage = 'disk'
            disk_args.hera_state_dir = str(tmp/'states')
            disk = self.run_training(disk_args, tmp/'disk')
            assert_states(memory, disk)
            self.assertEqual(memory.rs_test_acc, disk.rs_test_acc)
            for c in disk.clients:
                self.assertIsNone(c.model)
                self.assertIsNone(c.master)
                self.assertFalse(c.moments)
                self.assertTrue(c.state_path.exists())
                self.assertFalse(hasattr(c, 'validation_data'))
            # Resume a memory checkpoint into disk mode; storage is not algorithm config.
            first_args = copy.deepcopy(args)
            first_args.global_rounds = 1
            first = self.run_training(first_args, tmp/'first')
            resume_args = copy.deepcopy(disk_args)
            resume_args.hera_resume = str(tmp/'first'/'checkpoint')
            resumed = HERAFL(resume_args, 0)
            assert_states(first, resumed)
            resumed.result_directory = tmp/'resumed'
            with patch.object(resumed, 'save_results'), patch.object(resumed, 'cleanup_training_artifacts'):
                resumed.train()
            assert_states(memory, resumed)
            self.assertEqual(memory.rs_test_acc, resumed.rs_test_acc)

    def test_disk_failure_releases_tensors_and_preserves_saved_state(self):
        with self.environment() as tmp:
            server = HERAFL(self.args(hera_client_storage='disk', hera_state_dir=str(tmp)), 0)
            client = server.clients[0]
            before = client.state_path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, 'training failed'):
                with client.resident():
                    with torch.no_grad():
                        next(client.master.parameters()).add_(1)
                    raise RuntimeError('training failed')
            self.assertIsNone(client.master)
            self.assertIsNone(client.model)
            self.assertEqual(client.state_path.read_bytes(), before)
            with self.assertRaisesRegex(OSError, 'disk full'):
                with patch('torch.save', side_effect=OSError('disk full')), client.resident():
                    pass
            self.assertIsNone(client.master)
            self.assertEqual(client.state_path.read_bytes(), before)
            self.assertFalse(client.state_path.with_suffix('.tmp').exists())

    def test_validation_partition_is_not_retained(self):
        import gc
        import weakref
        with self.environment():
            server = HERAFL(self.args(), 0)
            client = server.clients[0]
            loader = client.load_train_data()
            self.assertFalse(hasattr(client, 'validation_data'))
            self.assertEqual(len(loader.dataset), client.train_samples)
            # A separately allocated partition must disappear with its loader.
            with patch('flcore.clients.clientbase.read_client_data',
                       side_effect=lambda *a, **kw: [(torch.randn(4), torch.tensor(i % 3)) for i in range(18)]):
                loader = client.load_train_data()
                sample = weakref.ref(loader.dataset.dataset[0][0])
                del loader
                gc.collect()
                self.assertIsNone(sample())
                client.local_accuracy_metrics()
                self.assertFalse(hasattr(client, 'validation_data'))

    def test_checkpoint_links_survive_cache_updates_and_removal(self):
        with self.environment() as tmp:
            args = self.args(hera_client_storage='disk', hera_state_dir=str(tmp/'states'), global_rounds=1)
            args.model.optimizer_kind = 'adam'
            server = self.run_training(args, tmp/'run')
            root = tmp/'run'/'checkpoint'
            generation = root/json.loads((root/'latest.json').read_text())['directory']
            client = server.clients[0]
            tensor_file = generation/'client_0_tensors.pt'
            self.assertEqual(tensor_file.stat().st_ino, client.state_path.stat().st_ino)
            saved = torch.load(tensor_file, map_location='cpu', weights_only=False)
            expected = next(saved['master'].parameters()).detach().clone()
            with client.resident():
                with torch.no_grad():
                    next(client.master.parameters()).add_(1.)
            self.assertNotEqual(tensor_file.stat().st_ino, client.state_path.stat().st_ino)
            server.state_directory.cleanup()
            resumed_args = copy.deepcopy(args)
            resumed_args.hera_resume = str(root)
            resumed = HERAFL(resumed_args, 0)
            with resumed.clients[0].resident(write=False):
                torch.testing.assert_close(next(resumed.clients[0].master.parameters()), expected)
                self.assertTrue(resumed.clients[0].moments)

    def test_failed_checkpoint_preserves_previous_generation(self):
        with self.environment() as tmp:
            server = self.run_training(self.args(hera_client_storage='disk', hera_state_dir=str(tmp/'states'), global_rounds=1), tmp/'run')
            root = tmp/'run'/'checkpoint'
            previous = (root/'latest.json').read_bytes()
            entries = set(root.iterdir())
            with patch('torch.save', side_effect=RuntimeError('file write failed')):
                with self.assertRaisesRegex(RuntimeError, 'incomplete generation was removed'):
                    server.save_checkpoint(root)
            self.assertEqual((root/'latest.json').read_bytes(), previous)
            self.assertEqual(set(root.iterdir()), entries)
            self.assertTrue(all(c.state_path.exists() for c in server.clients))

    def test_checkpoint_copies_when_linking_is_unavailable(self):
        import errno
        with self.environment() as tmp:
            server = HERAFL(self.args(hera_client_storage='disk', hera_state_dir=str(tmp/'states')), 0)
            root = tmp/'checkpoint'
            with patch('os.link', side_effect=OSError(errno.EXDEV, 'different filesystem')):
                server.save_checkpoint(root)
            generation = root/json.loads((root/'latest.json').read_text())['directory']
            source = server.clients[0].state_path
            target = generation/'client_0_tensors.pt'
            self.assertNotEqual(source.stat().st_ino, target.stat().st_ino)
            self.assertEqual(source.read_bytes(), target.read_bytes())

    def test_successful_export_cleans_only_current_run(self):
        with self.environment() as tmp:
            server = HERAFL(self.args(hera_client_storage='disk', hera_state_dir=str(tmp/'states'),
                                     hera_keep_checkpoints=False, hera_checkpoint_every=1, global_rounds=1), 0)
            server.result_directory = tmp/'run'
            cache = Path(server.state_directory.name)
            other = tmp/'other-run'/'checkpoint'
            other.mkdir(parents=True)
            (other/'keep.pt').write_bytes(b'other experiment')
            def export():
                self.assertTrue(cache.exists())
                self.assertTrue((server.result_directory/'checkpoint/latest.json').exists())
                (server.result_directory/'plots').mkdir()
                (server.result_directory/'plots/plot.png').write_bytes(b'plot')
                (server.result_directory/'metrics.h5').write_bytes(b'metrics')
            with patch.object(server, 'save_results', side_effect=export), patch.object(server, 'save_checkpoint', wraps=server.save_checkpoint) as save:
                server.train()
            self.assertEqual(save.call_count, 1)  # periodic only, no final duplicate
            self.assertFalse(cache.exists())
            self.assertFalse((server.result_directory/'checkpoint').exists())
            self.assertTrue((server.result_directory/'plots/plot.png').exists())
            self.assertTrue((server.result_directory/'metrics.h5').exists())
            self.assertTrue((other/'keep.pt').exists())
            server.cleanup_training_artifacts()  # idempotent

    def test_keep_checkpoints_releases_cache_and_allows_resume(self):
        with self.environment() as tmp:
            args = self.args(hera_client_storage='disk', hera_state_dir=str(tmp/'states'), global_rounds=1)
            server = HERAFL(args, 0)
            server.result_directory = tmp/'run'
            cache = Path(server.state_directory.name)
            with patch.object(server, 'save_results'):
                server.train()
            self.assertFalse(cache.exists())
            args.hera_resume = str(tmp/'run'/'checkpoint')
            resumed = HERAFL(args, 0)
            self.assertEqual(resumed.current_round, 1)
            self.assertEqual([c.trained_rounds for c in server.clients], [c.trained_rounds for c in resumed.clients])
            with resumed.clients[0].resident(write=False):
                self.assertIsNotNone(resumed.clients[0].master)

    def test_export_failure_preserves_training_artifacts(self):
        with self.environment() as tmp:
            server = HERAFL(self.args(hera_client_storage='disk', hera_state_dir=str(tmp/'states'),
                                     hera_keep_checkpoints=False, hera_checkpoint_every=1, global_rounds=1), 0)
            server.result_directory = tmp/'run'
            cache = Path(server.state_directory.name)
            with patch.object(server, 'save_results', side_effect=OSError('plot export failed')):
                with self.assertRaisesRegex(OSError, 'plot export failed'):
                    server.train()
            self.assertTrue(cache.exists())
            self.assertTrue((server.result_directory/'checkpoint/latest.json').exists())

    def test_result_schema_and_real_rendering(self):
        with self.environment() as tmp:
            server = self.run_training(self.args(global_rounds=1), tmp)
            server.save_results()
            import h5py
            with h5py.File(next(tmp.glob('*.h5'))) as f:
                self.assertIn('rs_test_acc', f)
                self.assertNotIn('rs_local_test_acc', f)
                self.assertEqual(f.attrs['evaluation_type'], 'global_model')
                for name, key in (('rs_train_loss', 'train_loss'), ('rs_test_auc', 'test_auc'),
                                  ('rs_test_acc_std', 'test_client_std'), ('rs_test_auc_std', 'test_auc_std')):
                    np.testing.assert_allclose(f[name][:], [server.round_reports[0][key]])
            self.assertEqual(len(list((tmp/'plots').glob('*.png'))), 9)
            self.assertTrue((tmp/'global_model.pt').exists())
            for metric in ('accuracy', 'train_loss', 'test_auc'):
                self.assertTrue((tmp/'plots'/f'global_{metric}_per_round_comparison.png').exists())
            for name in ('local_test_auc_per_round_comparison', 'local_test_loss_per_round_comparison', 'local_test_accuracy_per_round_comparison'):
                self.assertTrue((tmp/'plots'/f'{name}.png').exists(), name)
            for name in ('total_communication_per_experiment_comparison',
                         'total_energy_per_experiment_comparison', 'total_round_time_per_experiment_comparison'):
                self.assertTrue((tmp/'plots'/f'{name}.png').exists(), name)
            self.assertEqual(json.loads((tmp/'config.json').read_text())['evaluation_type'], 'global_model')


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
