"""HERAFL global weights + prototypes, with the official FedSC local losses.

JSON/base64 float32 transport sizes include serialization overhead. Local losses follow ../FedSC at a04dc8e630706a0c3f12f716c45857ead95a4940,
which differs from the PDF (no explicit U, CPDR uses MSE).
"""
import base64
import copy
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from .semantic import Bank, Payload, build_bank, SCHEMA

PROTOCOL = 'herafl-global-prototypes-v3'
FEDSC_REFERENCE = 'a04dc8e630706a0c3f12f716c45857ead95a4940'


def encode(value):
    def convert(v):
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().float().contiguous().numpy().astype('<f4')
            if not np.isfinite(a).all():
                raise ValueError('Nonfinite wire tensor.')
            return {'f32': base64.b64encode(a.tobytes()).decode('ascii'), 'shape': list(a.shape)}
        if isinstance(v, dict):
            return {str(k): convert(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [convert(x) for x in v]
        return v
    return json.dumps(convert(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def state_wire_size(state):
    """Exact encode(state) length without allocating its base64 weight payload."""
    metadata = {k: {'f32': '', 'shape': list(v.shape)} for k, v in state.items()}
    return len(encode(metadata)) + sum(4*((v.numel()*4+2)//3) for v in state.values())


def decode(data):
    def hook(v):
        if set(v) == {'f32', 'shape'}:
            a = np.frombuffer(base64.b64decode(v['f32'], validate=True), dtype='<f4').copy().reshape(v['shape'])
            if not np.isfinite(a).all():
                raise ValueError('Nonfinite wire tensor.')
            return torch.from_numpy(a)
        return v
    return json.loads(data, object_hook=hook)


def payload_dict(payload):
    return vars(payload)


def read_payload(value, dimension, classes, round_index, client_id):
    p = dict(value)
    for key in ('counts', 'prototypes'):
        p[key] = {int(k): v for k, v in p[key].items()}
    p = Payload(**p)
    if (p.version != SCHEMA or p.dimension != dimension or p.round != round_index
            or p.client_id != client_id or not 0 < p.retention <= 1):
        raise ValueError('Incompatible prototype payload identity/schema/dimension/round.')
    if set(p.prototypes) != set(p.counts):
        raise ValueError('Prototype support mismatch.')
    for k, v in p.prototypes.items():
        if not 0 <= k < classes or type(p.counts[k]) is not int or p.counts[k] <= 0:
            raise ValueError('Invalid class/support.')
        if not isinstance(v, torch.Tensor) or v.shape != (dimension,) or not torch.isfinite(v).all():
            raise ValueError('Invalid prototype vector.')
    if set(p.indicators) != set('HUDQ') or set(p.valid) != set('HUDQ'):
        raise ValueError('Invalid indicator schema.')
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in p.indicators.values()):
        raise ValueError('Invalid semantic indicator.')
    return p


def empty_reference(dimension):
    return dict(round=0, dimension=dimension, prototypes={}, relations={}, contributors={})


def reference_bank(reference):
    return Bank(reference['round'], reference['dimension'],
                {int(k): v for k, v in reference['prototypes'].items()}, reference['contributors'])


def build_reference(payloads, previous, round_index, classes, kind='simple', neighbors=1, device=None):
    if not payloads:
        return copy.deepcopy(previous)
    if kind == 'simple':
        bank = build_bank(payloads, reference_bank(previous), round_index)
        return dict(round=round_index, dimension=bank.dimension, prototypes=bank.prototypes,
                    relations={}, contributors=bank.contributors)
    # Match Global.get_re_p/get_co_p, including observable upstream behavior:
    # get_re_p changes a temporary stack but returns the original prototype lists.
    # get_co_p weights each class list by POSITION, not its contributor's ID.
    sizes = [sum(p.counts.values()) for p in payloads]
    if any(n <= 0 for n in sizes):
        raise ValueError('Official FedSC discrepancy weights require nonempty clients.')
    discrepancies = [math.sqrt(.5*sum((p.counts.get(k, 0)/n-1/classes)**2
                                     for k in range(classes))) for p, n in zip(payloads, sizes)]
    total, dis_total = sum(sizes), sum(discrepancies)
    if dis_total == 0:
        raise ValueError('Official FedSC divides by zero when every client has a uniform class distribution; no replacement weighting was invented.')
    raw = np.array([1/(1+np.exp(-(n/total-d/dis_total))) for n, d in zip(sizes, discrepancies)])
    weights = torch.as_tensor(raw/raw.sum(), device=device or next(iter(payloads[0].prototypes.values())).device)
    result = empty_reference(payloads[0].dimension)
    result['round'] = round_index
    members = {}
    for payload in payloads:
        for label, vector in payload.prototypes.items():
            members.setdefault(label, []).append(vector.detach().to(weights.device))
            result['contributors'].setdefault(label, []).append(payload.client_id)
    for label, vectors in members.items():
        weighted = torch.stack([v*weights[i] for i, v in enumerate(vectors)])
        # The weighted relations themselves are broadcast by the upstream code.
        result['relations'][label] = weighted.cpu()
        result['prototypes'][label] = weighted.sum(dim=0).cpu()
    return result


def semantic_losses(features, labels, reference, temperature=.05, compute_contrastive=True):
    """Match Local.local_train + InfoNCE_LCL in FedSC commit a04dc8e.

    CPDR is mean squared error on raw features. RPCL has one averaged positive
    per sample and the shared negative pool assembled across the minibatch,
    including its repetitions. No dataset-wide U is computed. References are
    fixed targets; the official missing-class fallback is the detached feature.
    """
    zero = features.sum()*0
    if not reference['prototypes'] and not reference['relations']:
        return zero, zero
    prototypes = {int(k): v.detach().to(features.device) for k, v in reference['prototypes'].items()}
    relations = {int(k): v.detach().to(features.device) for k, v in reference['relations'].items()}
    positive, targets, negatives = [], [], []
    for i, label in enumerate(labels.tolist()):
        if label in relations and label in prototypes:
            positive.append(relations[label].mean(dim=0))
            targets.append(prototypes[label])
        elif not relations and label in prototypes:  # simple-prototype ablation
            positive.append(prototypes[label])
            targets.append(prototypes[label])
        else:
            positive.append(features[i].detach())
            targets.append(features[i].detach())
        if compute_contrastive:
            negatives.extend(value for key, value in relations.items() if key != label)
    cpdr = F.mse_loss(features, torch.stack(targets))
    if not compute_contrastive or not relations:
        return cpdr, zero
    if not negatives:
        raise ValueError('FedSC RPCL requires a nonempty negative pool; the official implementation cannot stack an empty pool.')
    query = F.normalize(features, dim=-1)
    pos_key = F.normalize(torch.stack(positive), dim=-1)
    neg_keys = F.normalize(torch.cat(negatives, dim=0), dim=-1)
    pos_logit = torch.sum(query*pos_key, dim=1, keepdim=True)
    neg_logits = (query.unsqueeze(1) @ neg_keys.transpose(-2, -1)).squeeze(1)
    logits = torch.cat([pos_logit, neg_logits], dim=1)
    contrastive = F.cross_entropy(logits/temperature, torch.zeros(len(features), dtype=torch.long, device=features.device))
    return cpdr, contrastive


def resource_stats(telemetry, upload_bytes, download_bytes, profile):
    t = telemetry
    throughput = t['system_state']['throughput_mbps']
    down, up = download_bytes*8/(throughput*1e6), upload_bytes*8/(throughput*1e6)
    compute = sum(t[k] for k in ('training_time', 'semantic_extraction_time', 'adaptation_time', 'validation_time'))
    ce = profile.get('e_cmp_w', profile.get('e_cmp', .5))*compute
    de = profile.get('e_down_w', profile.get('e_down', .5))*down
    ue = profile.get('e_up_w', profile.get('e_up', .5))*up
    return dict(compute_time=compute, training_time=t['training_time'], semantic_extraction_time=t['semantic_extraction_time'],
                adaptation_time=t['adaptation_time'], validation_time=t['validation_time'],
                download_time=down, upload_time=up, round_time=compute+down+up,
                compute_energy=ce, download_energy=de, upload_energy=ue, energy_consumption=ce+de+ue,
                download_bytes=download_bytes, upload_bytes=upload_bytes, total_communication_bytes=download_bytes+upload_bytes,
                parameter_download_bytes=0, parameter_upload_bytes=0, throughput_mbps=throughput,
                active_parameter_bytes=t['active_parameter_bytes'], local_state_bytes=t['local_state_bytes'],
                parameter_fraction=t['parameter_fraction'], energy_source='profile_model', communication_source='serialized_protocol')
