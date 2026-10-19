"""Prototype-only wire protocol and FedSC-inspired, explicitly adapted losses.

Wire format is canonical UTF-8 JSON with base64 little-endian float32 vectors.
It contains no parameter/state-dict message type. Transport sizes include JSON
and base64 overhead, not just tensor storage. The FedSC U factor is not used;
CPDR here is normalized MSE, not a literal reproduction of Eq. 15.
"""
import base64
import copy
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from .semantic import Bank, Payload, build_bank, SCHEMA

PROTOCOL = 'herafl-prototypes-v1'


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


def build_reference(payloads, previous, round_index, classes, kind='simple', neighbors=1):
    if not payloads:
        return copy.deepcopy(previous)
    bank = build_bank(payloads, reference_bank(previous), round_index)
    result = dict(round=round_index, dimension=bank.dimension, prototypes=bank.prototypes,
                  relations={}, contributors=bank.contributors)
    if kind == 'simple':
        return result
    # Discrepancy weights indexed by real client IDs, never by class-list position.
    sizes = {p.client_id: sum(p.counts.values()) for p in payloads}
    discrepancies = {p.client_id: math.sqrt(.5*sum((p.counts.get(k, 0)/max(sizes[p.client_id], 1)-1/classes)**2
                                                  for k in range(classes))) for p in payloads}
    total, dis_total = sum(sizes.values()), sum(discrepancies.values())
    weights = {i: 1/(1+math.exp(-(sizes[i]/max(total, 1) - (discrepancies[i]/dis_total if dis_total else 0)))) for i in sizes}
    for label in bank.prototypes:
        members = sorted([(p.client_id, p.prototypes[label]) for p in payloads if label in p.prototypes])
        center = torch.stack([v for _, v in members]).mean(0)
        valid = {i: v for i, v in members if v.norm() > 1e-12 and center.norm() > 1e-12}
        angular = {i: float(F.cosine_similarity(v, center, dim=0)) for i, v in valid.items()}
        relations, ids = [], []
        for i, vector in members:
            nearest = sorted((j for j in valid if j != i), key=lambda j: (abs(angular[i]-angular[j]), j))[:neighbors] if i in valid else []
            relations.append(torch.stack([vector] + [valid[j] for j in nearest]).mean(0))
            ids.append(i)
        w = torch.tensor([weights[i] for i in ids])
        result['relations'][label] = torch.stack(relations)
        result['prototypes'][label] = (torch.stack(relations)* (w/w.sum())[:, None]).sum(0)
    return result


def semantic_losses(features, labels, reference, temperature=.05):
    """Normalized alignment/CPDR and multi-positive cosine InfoNCE."""
    zero = features.sum()*0
    alignment, contrastive = [], []
    prototypes = {int(k): v.detach().to(features.device) for k, v in reference['prototypes'].items()}
    relations = {int(k): v.detach().to(features.device) for k, v in reference['relations'].items()}
    z = F.normalize(features, dim=1)
    for label in labels.unique().tolist():
        subset = z[labels == label]
        if label in prototypes and prototypes[label].norm() > 1e-12:
            target = F.normalize(prototypes[label], dim=0)
            alignment.append((subset-target).square().mean(1))
        positive = relations.get(label)
        negatives = [v for k, v in relations.items() if k != label]
        if positive is None or not negatives:
            continue
        positive = positive[positive.norm(dim=1) > 1e-12]
        negative = torch.cat(negatives)
        negative = negative[negative.norm(dim=1) > 1e-12]
        if len(positive) and len(negative):
            pos = subset @ F.normalize(positive, dim=1).T / temperature
            neg = subset @ F.normalize(negative, dim=1).T / temperature
            contrastive.append(torch.logsumexp(torch.cat([pos, neg], dim=1), dim=1)-torch.logsumexp(pos, dim=1))
    return (torch.cat(alignment).mean() if alignment else zero,
            torch.cat(contrastive).mean() if contrastive else zero)


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
