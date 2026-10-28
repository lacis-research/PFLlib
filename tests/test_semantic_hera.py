"""Deterministic semantic selection and complete native CPU lifecycle checks."""
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'system'))
from flcore.heterogeneity.semantic import (Bank, Payload, extract, build_bank, history_context,
                                          wire_bytes, SemanticLinUCB, summarize_confusion)
from flcore.heterogeneity.pruning import prune_model, aggregate_submodels, recover_model
from flcore.trainmodel.models import DNN, FedAvgCNN, Mclr_Logistic
from flcore.servers.serverherafl import HERAFL
import test_heterogeneous_algorithms as native_tests


class FeatureTests(unittest.TestCase):
    def test_preserved_coordinates_physical_size_backward_and_full_path(self):
        torch.manual_seed(5)
        for model, x, name in [(DNN(4, 8, 3), torch.randn(3, 4), 'fc1'),
                               (FedAvgCNN(), torch.randn(3, 1, 28, 28), 'fc1.0')]:
            model.eval()
            for rate in (.25, .75, 1.):
                sub, coords = prune_model(model, rate, preserve_representation=True)
                out, features = sub.forward_with_features(x)
                torch.testing.assert_close(out, sub(x), rtol=0, atol=0)
                self.assertEqual(features.shape[1], model.fc.in_features)
                self.assertEqual(coords[name+'.bias'][0].tolist(), list(range(model.fc.in_features)))
                self.assertEqual(coords['fc.weight'][1].tolist(), list(range(model.fc.in_features)))
                out.sum().backward()
                self.assertTrue(all(p.grad is not None for p in sub.parameters()))
                if rate < 1:
                    self.assertLess(sum(p.numel() for p in sub.parameters()), sum(p.numel() for p in model.parameters()))
                else:
                    torch.testing.assert_close(out, model(x), rtol=0, atol=0)
                _, masks = recover_model(sub, coords, model)
                result = aggregate_submodels(model, [(sub, coords, 3)])
                for key, value in model.named_parameters():
                    torch.testing.assert_close(result.get_parameter(key)[~masks[key].bool()], value[~masks[key].bool()])

    def test_known_indicators_prototypes_null_vectors_and_bytes(self):
        model = DNN(2, 2, 2)
        with torch.no_grad():
            model.fc1.weight.copy_(torch.eye(2)); model.fc1.bias.zero_()
            model.fc.weight.zero_(); model.fc.bias.zero_()
        data = [(torch.tensor(x), torch.tensor(y)) for x, y in [([1., 0.], 0), ([3., 0.], 0), ([0., 2.], 1)]]
        bank = Bank(1, 2, {0: torch.tensor([0., 1.]), 1: torch.tensor([0., 2.])})
        model.train()
        payload = extract(model, DataLoader(data, batch_size=2), 2, 7, 2, .5, bank, 'cpu')
        self.assertTrue(model.training)
        self.assertEqual(payload.counts, {0: 2, 1: 1})
        torch.testing.assert_close(payload.prototypes[0], torch.tensor([2., 0.]))
        self.assertFalse(payload.prototypes[0].requires_grad)
        self.assertAlmostEqual(payload.indicators['H'], -(2/3*np.log2(2/3)+1/3*np.log2(1/3)))
        self.assertAlmostEqual(payload.indicators['U'], 1., places=6)
        self.assertAlmostEqual(payload.indicators['D'], 1/3, places=6)
        self.assertAlmostEqual(payload.indicators['Q'], 1-2/14, places=6)
        header = len(json.dumps(payload.metadata(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        self.assertEqual(wire_bytes(payload), 8+header+2*(8+2*4))
        zeros = [(torch.zeros(2), torch.tensor(0))]
        empty = extract(model, DataLoader(zeros), 2, 7, 2, .5, bank, 'cpu')
        self.assertEqual(empty.indicators['H'], 0)
        self.assertFalse(empty.valid['Q']); self.assertFalse(empty.valid['D'])

    def test_bank_weighting_missing_classes_no_alias_and_expiration(self):
        p = Payload(8, 2, .5, 2, {1: 2}, {1: torch.tensor([2., 0.])},
                    dict(H=.2,U=.3,D=.4,Q=.5), dict(H=True,U=True,D=False,Q=True), 0., 1)
        q = copy.deepcopy(p); q.client_id=3; q.counts={1: 1}; q.prototypes[1].fill_(5)
        old = Bank(1, 2, {0: torch.zeros(2)})
        bank = build_bank([p,q], old, 2)
        torch.testing.assert_close(bank.prototypes[1], torch.tensor([3., 5/3]))
        self.assertNotIn(0, bank.prototypes)
        self.assertIs(build_bank([], bank, 3), bank)
        p.prototypes[1].zero_(); self.assertEqual(float(bank.prototypes[1][0]), 3.)
        self.assertEqual(history_context(p,2,5,'HUDQ')['semantic_available'], 0)
        self.assertEqual(history_context(p,3,5,'HUDQ')['semantic_available'], 1)
        self.assertEqual(history_context(p,8,5,'HUDQ')['semantic_available'], 0)
        self.assertEqual(history_context(p,3,5,'H')['semantic_valid_U'], 0)
        self.assertEqual(history_context(p,3,5,'H')['semantic_coverage'], 0)
        self.assertEqual(history_context(p,3,5,'D')['semantic_available'], 0)
        summary=summarize_confusion(torch.tensor([[2,0],[0,0]]))
        self.assertEqual(summary['balanced_accuracy'],1.); self.assertIsNone(summary['recall'][1])


class SelectionTests(unittest.TestCase):
    def policy(self, kind='linucb'):
        p=SemanticLinUCB([.25,.5,1.], alpha=0)
        p.enable(kind, semantic_weight=2., cost_weight=0.)
        return p

    def context(self, uncertainty):
        c=dict(f_i=.5, c_r_i=.5, dataset_size=1.)
        p=Payload(1,1,.5,2,{0:1},{},dict(H=.5,U=uncertainty,D=0.,Q=1.),dict.fromkeys('HUDQ',True),1.,0)
        c.update(history_context(p,2,5,'HUDQ'))
        return c

    def test_semantics_change_actions_same_resources_and_reproducible(self):
        costs={a:dict(fraction=a,feasible=True) for a in (.25,.5,1.)}
        for kind in ('heuristic','linucb'):
            p=self.policy(kind)
            if kind=='linucb':
                # Controlled arm-specific coefficients: reward of large model depends on U.
                p.b[.25][p.feature_order.index('dataset_size')]=.2
                p.b[1.][p.feature_order.index('semantic_U')]=1.
            low=p.choose_semantic(self.context(0.),[],costs,False)[0]
            high=p.choose_semantic(self.context(1.),[],costs,False)[0]
            self.assertNotEqual(low,high)
            self.assertEqual(high,p.choose_semantic(self.context(1.),[],costs,False)[0])

    def test_physical_limits_and_empty_feasible_without_capacity_floors(self):
        p=self.policy()
        context=self.context(.8); context.update(static_cpu_capacity=1.,static_compute_cost=0.)
        costs={a:dict(fraction=a,feasible=a==.25) for a in p.pruning_rates}
        arm,info=p.choose_semantic(context,[],costs,False)
        self.assertEqual(arm,.25); self.assertNotIn('retention_floor_satisfied',info)
        for v in costs.values(): v['feasible']=False
        self.assertIsNone(p.choose_semantic(context,[],costs,False)[0])

    def test_schema_rejects_old_checkpoint(self):
        p=self.policy()
        with self.assertRaisesRegex(ValueError,'schema'):
            p.load_state_dict({'schema':'legacy'})
        p.load_state_dict(p.state_dict())


class LifecycleTests(unittest.TestCase):
    make_args=native_tests.NativeServerTests.make_args
    run_server=native_tests.NativeServerTests.run_server

    def test_global_alignment_changes_learning_and_evaluates_global_weights(self):
        evaluated = []
        evaluate = HERAFL.evaluate
        def check_global(server, *args, **kwargs):
            self.assertTrue(all(c.model is server.global_model for c in server.clients))
            evaluated.append(server.current_round)
            return evaluate(server, *args, **kwargs)
        servers = []
        for weight in (0., 1.):
            torch.manual_seed(123)
            np.random.seed(123)
            args = self.make_args(algorithm='HERAFL', hera_semantic=True,
                                  hera_global_align_weight=weight, global_rounds=2,
                                  wait_for_all=True, aligned_comparison=True,
                                  reward_validation_fraction=.2, detailed_evaluation=True)
            with patch.object(HERAFL, 'evaluate', check_global):
                server, _ = self.run_server(HERAFL, args)
            servers.append(server)
        self.assertEqual(evaluated, [1, 2, 1, 2])
        for left, right in zip(servers[0].round_reports[0]['clients'], servers[1].round_reports[0]['clients']):
            self.assertEqual(left['accuracy'], right['accuracy'])
            self.assertEqual(right['round_stats']['prototype_alignment_samples'], 0)
        self.assertTrue(any(r['round_stats']['prototype_alignment_samples'] > 0
                            for r in servers[1].round_reports[1]['clients']))
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(servers[0].global_model.parameters(), servers[1].global_model.parameters())))
        self.assertTrue(all(r['global_accuracy'] is not None for r in servers[1].round_reports))
        self.assertIn('balanced_accuracy', servers[1].round_reports[-1]['test_sample_weighted'])

    def test_global_alignment_expires_references_and_collects_with_legacy_policy(self):
        server, _ = self.run_server(HERAFL, self.make_args(
            algorithm='HERAFL', hera_semantic=True, hera_global_align_weight=1.,
            hera_semantic_policy='legacy', hera_semantic_max_age=0,
            global_rounds=2, wait_for_all=True))
        self.assertTrue(server.semantic_collect)
        self.assertTrue(server.semantic_bank.prototypes)
        self.assertTrue(all(r['round_stats']['prototype_alignment_samples'] == 0
                            for rr in server.round_reports for r in rr['clients']))

    def test_global_alignment_validates_configuration(self):
        for weight in (-1., float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                HERAFL(self.make_args(algorithm='HERAFL', hera_semantic=True,
                                     hera_global_align_weight=weight), 0)
        with self.assertRaisesRegex(ValueError, 'requires --hera-semantic'):
            HERAFL(self.make_args(algorithm='HERAFL', hera_global_align_weight=1.), 0)

    def test_alignment_has_feature_gradients_and_detached_targets(self):
        from flcore.clients.clientherafl import clientHERAFL
        client = object.__new__(clientHERAFL)
        client.model = DNN(4, 8, 3)
        client.loss = torch.nn.CrossEntropyLoss()
        client.global_align_weight = 1.
        teacher = torch.randn(8, requires_grad=True)
        client.alignment_targets = {0: torch.nn.functional.normalize(teacher.detach(), dim=0)}
        client.alignment_sum, client.alignment_samples = 0., 0
        x, y = torch.randn(3, 4), torch.tensor([0, 0, 2])
        loss = client.training_loss(x, y)
        ce = client.loss(client.model(x), y)
        self.assertGreater(float(loss-ce), 0.)
        (loss-ce).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in client.model.parameters()))
        self.assertIsNone(teacher.grad)
        self.assertEqual(client.alignment_samples, 2)
        torch.testing.assert_close(client.training_loss(x, torch.full_like(y, 1)),
                                   client.loss(client.model(x), torch.full_like(y, 1)))

    def test_two_rounds_frozen_context_previous_bank_costs_and_validation(self):
        args=self.make_args(algorithm='HERAFL', hera_semantic=True, global_rounds=2,
                            pruning_arms=[.5,1.],
                            wait_for_all=True, aligned_comparison=True,
                            reward_validation_fraction=.2, detailed_evaluation=True)
        actual=HERAFL.assign_submodel
        def alternating(server,client,context,pending):
            # Force different arms through candidate feasibility, exercising actual assignment path.
            wanted=.5 if client.id%2 else 1.
            original=server.bandit.choose_semantic
            def choose(c,r,costs,f):
                for a in costs: costs[a]['feasible']=a==wanted
                return original(c,r,costs,f)
            with patch.object(server.bandit,'choose_semantic',choose):
                return actual(server,client,context,pending)
        with patch.object(HERAFL,'assign_submodel',alternating):
            server,original=self.run_server(HERAFL,args)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(original.parameters(),server.global_model.parameters())))
        self.assertEqual(server.semantic_bank.round,2)
        self.assertEqual(set(r['retention'] for rr in server.round_reports for r in rr['clients']),{.5,1.})
        for rr in server.round_reports:
            self.assertIn('test_sample_weighted',rr)
            for r in rr['clients']:
                self.assertEqual(r['learning_context'],r['selection_context'])
                self.assertEqual(r['semantic_payload']['reference_round'],rr['round']-1)
                self.assertEqual(sum(r['semantic_payload']['counts'].values()),15)
                stats=r['round_stats']
                self.assertEqual(stats['total_communication_bytes'],stats['parameter_upload_bytes']+stats['parameter_download_bytes']+stats['semantic_upload_bytes']+stats['semantic_download_bytes']+stats.get('coordinate_download_bytes',0))
                self.assertGreater(stats['compute_time'],stats['training_time'])
        for arm in server.bandit.pruning_rates:
            expected=np.eye(server.bandit.dim)
            for rr in server.round_reports:
                for r in rr['clients']:
                    if r['retention']==arm:
                        x=server.bandit.vectorize_context(r['selection_context']); expected+=np.outer(x,x)
            np.testing.assert_allclose(server.bandit.A[arm],expected)

    def test_discarded_infeasible_and_legacy_control(self):
        for options in (dict(pruning_round_timeout=1e-12),dict(hera_semantic_memory_limit=1.)):
            server,original=self.run_server(HERAFL,self.make_args(algorithm='HERAFL',hera_semantic=True,global_rounds=2,**options))
            self.assertEqual(server.semantic_bank.round,0); self.assertFalse(server.semantic_history)
            for a,b in zip(original.parameters(),server.global_model.parameters()): torch.testing.assert_close(a,b)
        server,_=self.run_server(HERAFL,self.make_args(algorithm='HERAFL',hera_semantic=True,hera_semantic_policy='legacy'))
        self.assertFalse(server.semantic_collect)
        self.assertNotIn('semantic_upload_bytes',server.round_reports[0]['clients'][0]['round_stats'])

    def test_empty_second_round_preserves_bank_and_parameters(self):
        accepted_state = {}
        original = HERAFL.after_aggregation
        def after(server, responses, reports):
            result = original(server, responses, reports)
            if server.current_round == 1:
                accepted_state['bank'] = copy.deepcopy(server.semantic_bank)
                accepted_state['model'] = copy.deepcopy(server.global_model.state_dict())
            return result
        def timeout(server, contexts):
            return 300 if server.current_round == 1 else 1e-12
        with patch.object(HERAFL, 'after_aggregation', after), patch.object(HERAFL, 'round_timeout', timeout):
            server, _ = self.run_server(HERAFL, self.make_args(algorithm='HERAFL', hera_semantic=True, global_rounds=2))
        self.assertEqual(server.semantic_bank.round, 1)
        self.assertEqual(server.round_reports[-1]['semantic_bank_age'], 1)
        for k,v in accepted_state['model'].items():
            torch.testing.assert_close(server.global_model.state_dict()[k], v)
        for k,v in accepted_state['bank'].prototypes.items():
            torch.testing.assert_close(server.semantic_bank.prototypes[k], v)
        self.assertTrue(all(p.round == 1 for p in server.semantic_history.values()))

    def test_semantic_overhead_participates_in_timeout(self):
        from flcore.clients.clientherafl import clientHERAFL
        original = clientHERAFL.record_resources
        # Make base training time small and post-extraction time large deterministically.
        calls = {}
        def measured(client, compute_time):
            calls[client.id] = calls.get(client.id, 0) + 1
            original(client, .001 if calls[client.id] == 1 else 10.)
        with patch.object(clientHERAFL, 'record_resources', measured):
            server, _ = self.run_server(HERAFL, self.make_args(algorithm='HERAFL', hera_semantic=True,
                                                             aligned_comparison=True, pruning_round_timeout=1.))
        for r in server.round_reports[0]['clients']:
            self.assertEqual(r['status'], 'timeout')
            self.assertLess(r['round_stats']['training_time'],1.)
            self.assertGreater(r['round_stats']['round_time'],10.)
        self.assertFalse(server.semantic_bank.prototypes)

    def test_save_infeasible_and_semantic_results(self):
        import tempfile
        from flcore.heterogeneity.results import save_policy_results
        for extra in ({}, {'hera_semantic_memory_limit':1.}):
            server,_=self.run_server(HERAFL,self.make_args(algorithm='HERAFL',hera_semantic=True,**extra))
            with tempfile.TemporaryDirectory() as directory:
                with patch('flcore.heterogeneity.results.result_directory',return_value=Path(directory)):
                    save_policy_results(server)
                metadata=json.loads((Path(directory)/'config.json').read_text())
                self.assertEqual(metadata['semantic_schema'],'fedload-semantic-v2')
                reports=json.loads(next(Path(directory).glob('*_policy.json')).read_text())
                self.assertEqual(len(reports),1)

    def test_unsupported_model_fails_before_client_setup(self):
        with self.assertRaisesRegex(ValueError,'only FedAvgCNN, DNN, VGG11 and VGG16'):
            HERAFL(self.make_args(algorithm='HERAFL',hera_semantic=True,model=Mclr_Logistic()),0)
