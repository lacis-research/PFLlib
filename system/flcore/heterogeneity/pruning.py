"""Structured L1 pruning of native PFLlib models (PyTorch NCHW layout)."""
import copy
import math
import torch
from torch import nn
from flcore.trainmodel.models import FedAvgCNN, DNN, Mclr_Logistic


def validate_model(model):
    if type(model) not in (FedAvgCNN, DNN, Mclr_Logistic):
        raise ValueError('FedLoad/HERAFL support native CNN (FedAvgCNN), DNN and MLR models.')


def prune_model(model, retention):
    """Return a physically smaller model and parameter coordinates for recovery."""
    validate_model(model)
    if not math.isfinite(retention) or not 0 < retention <= 1:
        raise ValueError('Retention must be in (0, 1].')
    result = copy.deepcopy(model)
    layers = [(name, layer) for name, layer in model.named_modules()
              if isinstance(layer, (nn.Conv2d, nn.Linear))]
    coordinates = {}
    previous = None
    previous_layer = None
    for index, (name, layer) in enumerate(layers):
        weight = layer.weight.detach()
        outputs = torch.arange(weight.shape[0], device=weight.device)
        if index < len(layers) - 1:
            importance = weight.abs().flatten(1).sum(1)
            outputs = torch.argsort(importance, descending=True, stable=True)[
                :max(1, math.floor(len(outputs) * retention))].sort().values
        inputs = torch.arange(weight.shape[1], device=weight.device)
        if previous is not None:
            if isinstance(previous_layer, nn.Conv2d) and isinstance(layer, nn.Linear):
                # PyTorch flatten orders all spatial positions of each channel together.
                channels = previous_layer.out_channels
                if layer.in_features % channels:
                    raise ValueError('Conv-to-linear dimensions are not divisible by channels.')
                spatial = layer.in_features // channels
                inputs = (previous[:, None] * spatial + torch.arange(
                    spatial, device=weight.device)[None, :]).flatten()
            else:
                inputs = previous
        target = result.get_submodule(name)
        target.weight = nn.Parameter(weight.index_select(0, outputs).index_select(1, inputs).clone())
        coordinates[name + '.weight'] = (outputs, inputs)
        if layer.bias is not None:
            target.bias = nn.Parameter(layer.bias.detach().index_select(0, outputs).clone())
            coordinates[name + '.bias'] = (outputs,)
        if isinstance(target, nn.Conv2d):
            target.in_channels, target.out_channels = len(inputs), len(outputs)
        else:
            target.in_features, target.out_features = len(inputs), len(outputs)
        previous, previous_layer = outputs, layer
    return result, coordinates


def recover_model(submodel, coordinates, template):
    """Scatter trained values and coverage masks into the original dimensions."""
    values, masks = {}, {}
    for name, parameter in template.named_parameters():
        values[name] = torch.zeros_like(parameter)
        masks[name] = torch.zeros_like(parameter)
        axes = coordinates[name]
        if len(axes) == 2:
            indices = (axes[0][:, None], axes[1][None, :])
        else:
            indices = axes
        values[name][indices] = submodel.get_parameter(name).detach()
        masks[name][indices] = 1
    return values, masks


def aggregate_submodels(template, updates):
    """Sample-weighted average at covered coordinates, preserving uncovered values."""
    result = copy.deepcopy(template)
    numerator = {n: torch.zeros_like(p) for n, p in template.named_parameters()}
    denominator = {n: torch.zeros_like(p) for n, p in template.named_parameters()}
    for model, coordinates, samples in updates:
        if samples <= 0:
            continue
        values, masks = recover_model(model, coordinates, template)
        for name in numerator:
            numerator[name].add_(values[name], alpha=float(samples))
            denominator[name].add_(masks[name], alpha=float(samples))
    with torch.no_grad():
        for name, parameter in result.named_parameters():
            covered = denominator[name] > 0
            parameter[covered] = numerator[name][covered] / denominator[name][covered]
    return result
