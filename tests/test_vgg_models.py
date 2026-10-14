"""VGG layer counts, partial training and optimizer compatibility."""
import gc
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'system'))
from flcore.trainmodel.vgg import VGG11, VGG16
from flcore.heterogeneity.pruning import prune_model, aggregate_submodels
from flcore.clients.clientfedload import clientFedLoad


class VGGTests(unittest.TestCase):
    def test_architectures_and_pruned_training_preserve_output_classes(self):
        for cls, count in ((VGG11, 8), (VGG16, 13)):
            with self.subTest(model=cls.__name__):
                model = cls(num_classes=100)
                convs = [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]
                linear = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
                self.assertEqual(len(convs), count)
                self.assertEqual([(l.in_features, l.out_features) for l in linear], [(512, 4096), (4096, 4096), (4096, 100)])
                self.assertEqual(sum(isinstance(l, torch.nn.MaxPool2d) for l in model.modules()), 5)
                model.eval()
                x = torch.randn(2, 3, 32, 32)
                with torch.no_grad():
                    self.assertEqual(tuple(model(x).shape), (2, 100))
                submodel, coordinates = prune_model(model, 0.0625)
                self.assertLess(sum(p.numel() for p in submodel.parameters()), sum(p.numel() for p in model.parameters()))
                submodel.train()
                output = submodel(x)
                self.assertEqual(tuple(output.shape), (2, 100))
                torch.nn.functional.cross_entropy(output, torch.tensor([0, 99])).backward()
                self.assertTrue(all(p.grad is not None for p in submodel.parameters()))
                client = clientFedLoad.__new__(clientFedLoad)
                client.learning_rate = 0.001
                client.learning_rate_decay = False
                client.set_submodel(submodel, coordinates, 0.0625)
                self.assertIsInstance(client.optimizer, torch.optim.Adam)
                client.optimizer.step()
                del model, submodel, client, output
                gc.collect()

    def test_aggregation_does_not_build_dense_per_client_recovery_masks(self):
        from unittest.mock import patch
        from flcore.trainmodel.models import DNN
        model = DNN(input_dim=4, mid_dim=8, num_classes=3)
        submodel, coordinates = prune_model(model, 0.5, device='cpu')
        with patch('flcore.heterogeneity.pruning.recover_model', side_effect=AssertionError('dense recovery allocation')):
            result = aggregate_submodels(model, [(submodel, coordinates, 10)])
        for original, aggregated in zip(model.parameters(), result.parameters()):
            torch.testing.assert_close(original, aggregated)

    def test_evaluation_is_deterministic_without_augmentation_or_dropout(self):
        model = VGG11().eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
