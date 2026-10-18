"""HERAFL shares the FedLoad experiment base; only semantic extensions differ."""
import copy
import random
import unittest
from unittest.mock import patch
import numpy as np
import torch
import test_heterogeneous_algorithms as native
from flcore.clients.clientfedload import clientFedLoad
from flcore.servers.serverfedload import FedLoad
from flcore.servers.serverherafl import HERAFL
from flcore.heterogeneity.linucb_fedload import LinUCBBandit
from flcore.heterogeneity.semantic import SemanticLinUCB


class FedLoadBaseTests(unittest.TestCase):
    def test_disabled_semantics_matches_fedload_with_controlled_measurements(self):
        helper = native.NativeServerTests()
        record = clientFedLoad.record_resources
        def fixed_time(client, compute_time):
            record(client, .01)
        servers = []
        for cls in (FedLoad, HERAFL):
            torch.manual_seed(91)
            np.random.seed(91)
            random.seed(91)
            with patch.object(clientFedLoad, 'record_resources', fixed_time):
                server, _ = helper.run_server(cls, helper.make_args(
                    algorithm=cls.__name__, global_rounds=3, pruning_arms=[.25,.5,1.],
                    aligned_comparison=False, hera_semantic=False))
            servers.append(server)
        first, second = servers
        self.assertIs(type(second.bandit), LinUCBBandit)
        self.assertEqual(first.round_reports, second.round_reports)
        self.assertEqual([c.profile for c in first.clients], [c.profile for c in second.clients])
        self.assertEqual(first.rs_test_acc, second.rs_test_acc)
        for a,b in zip(first.global_model.parameters(), second.global_model.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        for arm in first.bandit.pruning_rates:
            np.testing.assert_array_equal(first.bandit.A[arm], second.bandit.A[arm])
            np.testing.assert_array_equal(first.bandit.b[arm], second.bandit.b[arm])
        self.assertFalse(hasattr(second,'predictor'))
        self.assertFalse(hasattr(second,'selector'))

    def test_semantic_mode_uses_fedload_selection_profiles_timeout_and_features(self):
        helper=native.NativeServerTests()
        torch.manual_seed(9); np.random.seed(9); random.seed(9)
        base,_=helper.run_server(FedLoad, helper.make_args(algorithm='FedLoad'))
        torch.manual_seed(9); np.random.seed(9); random.seed(9)
        semantic,_=helper.run_server(HERAFL, helper.make_args(algorithm='HERAFL',hera_semantic=True))
        self.assertIs(HERAFL.select_clients, FedLoad.select_clients)
        self.assertIs(HERAFL.round_timeout, FedLoad.round_timeout)
        self.assertEqual([c.profile for c in base.clients],[c.profile for c in semantic.clients])
        self.assertEqual(base.bandit.feature_order, semantic.bandit.feature_order[:semantic.bandit.base_dim])
        for r in semantic.round_reports[0]['clients']:
            self.assertFalse(any(k.startswith('static_') or k.startswith('straggler') or k=='predicted_round_time'
                                 for k in r['selection_context']))
            self.assertNotIn('capacity_group',r['alpha_info'])
            for score in r['alpha_info']['scores'].values():
                self.assertNotIn('capacity',score)

    def test_semantic_fallback_matches_fedload_and_old_schema_rejected(self):
        base=LinUCBBandit([.25,.5,1.],random_seed=7)
        policy=SemanticLinUCB([.25,.5,1.],random_seed=7)
        policy.legacy_policy=copy.deepcopy(policy)
        policy.enable()
        costs={a:dict(feasible=True,fraction=a) for a in base.pruning_rates}
        c=dict(f_i=.9,c_r_i=.1,dataset_size=.7)
        for _ in range(5):
            a,_=base.choose_pruning_rate(c,[])
            b,_=policy.choose_semantic(c,[],costs,True)
            self.assertEqual(a,b)
            base.update(a,.8,c); policy.legacy_policy.update(b,.8,c)
        old=policy.state_dict(); old['schema']='hera-semantic-v1'
        with self.assertRaisesRegex(ValueError,'schema'):
            policy.load_state_dict(old)
