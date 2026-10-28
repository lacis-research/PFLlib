#!/usr/bin/env python3
"""Bounded CUDA loss ablation on real training subsets; not a federated benchmark."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'system'))
from flcore.trainmodel.models import FedAvgCNN
from flcore.heterogeneity.prototype_protocol import semantic_losses, empty_reference, FEDSC_REFERENCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda'], required=True)
    parser.add_argument('--samples-per-client', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--output', type=Path, default=ROOT/'tmp/hera-loss-diagnostic-official.json')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required; no CPU fallback.')
    if args.samples_per_client < 1 or args.epochs < 1:
        parser.error('samples and epochs must be positive')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    xs, ys = [], []
    for cid in (0, 4):
        data = np.load(ROOT/f'dataset/Cifar10/train/{cid}.npz', allow_pickle=True)['data'].item()
        order = np.random.default_rng(42+cid).permutation(len(data['y']))
        # Exclude the same 20% validation prefix used by clientFedLoad.
        order = order[max(1, int(len(order)*.2)):][:args.samples_per_client]
        xs.append(torch.as_tensor(np.asarray(data['x'])[order], dtype=torch.float32, device='cuda'))
        ys.append(torch.as_tensor(np.asarray(data['y'])[order], dtype=torch.long, device='cuda'))
    x, y = torch.cat(xs), torch.cat(ys)
    loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=False)
    base = FedAvgCNN(in_features=3, num_classes=10, dim=1600).cuda()
    optimizer = torch.optim.SGD(base.parameters(), lr=.001)
    for bx, by in loader:  # identical short CE warmup for all variants
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(base(bx), by).backward()
        optimizer.step()
    # Controlled frozen class references, not a reconstruction of the interrupted run.
    with torch.no_grad():
        features = base.forward_features(x)
        prototypes = {int(k): features[y == k].mean(0) for k in y.unique()}
    reference = dict(empty_reference(512), prototypes=prototypes,
                     relations={k: v.unsqueeze(0) for k, v in prototypes.items()})
    result = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  fedsc_reference=FEDSC_REFERENCE, loss_definition='fedsc-official', seed=42, dtype='float32', clients=[0, 4], samples=len(x), batch_size=64,
                  scope='Training-only pooled subset; no pruning or federation; fixed class-mean references after shared CE warmup.',
                  variants={})
    for name, cpdr_weight, rpcl_weight in [('CE', 0., 0.), ('CE_CPDR', 1., 0.), ('CE_RPCL', 0., 1.), ('CE_CPDR_RPCL', 1., 1.)]:
        model = copy.deepcopy(base)
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        records = []
        for epoch in range(args.epochs):
            sums = np.zeros(4)
            for bx, by in loader:
                optimizer.zero_grad()
                logits, z = model.forward_with_features(bx)
                ce = torch.nn.functional.cross_entropy(logits, by)
                cpdr, rpcl = semantic_losses(z, by, reference, compute_contrastive=bool(rpcl_weight))
                loss = ce+cpdr_weight*cpdr+rpcl_weight*rpcl
                if not records and not sums.any():
                    norms = {}
                    for key, term in [('ce', ce), ('cpdr', cpdr), ('rpcl', rpcl)]:
                        gradients = torch.autograd.grad(term, tuple(model.parameters()), retain_graph=True, allow_unused=True)
                        norms[key] = float(torch.stack([g.square().sum() for g in gradients if g is not None]).sum().sqrt())
                    first_gradient_norms = norms
                loss.backward()
                optimizer.step()
                sums += np.array([float(t.detach()) for t in (ce, cpdr, rpcl, loss)])*len(by)
            torch.cuda.synchronize()
            with torch.no_grad():
                evaluation_ce = float(torch.nn.functional.cross_entropy(model(x), y))
                feature_norm = float(model.forward_features(x).norm(dim=1).mean())
            records.append(dict(epoch=epoch+1, **dict(zip(('ce', 'cpdr', 'rpcl', 'total'), (sums/len(x)).tolist())),
                                evaluation_train_ce=evaluation_ce, feature_norm=feature_norm))
        torch.cuda.synchronize()
        result['variants'][name] = dict(seconds=time.perf_counter()-start, epochs=records,
                                        initial_gradient_norms=first_gradient_norms,
                                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
        print(name, json.dumps(result['variants'][name]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(args.output)


if __name__ == '__main__':
    main()
