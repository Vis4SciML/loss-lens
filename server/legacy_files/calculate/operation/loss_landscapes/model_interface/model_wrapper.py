""" Class used to define interface to complex models """

import abc
import typing
import copy
import itertools
import torch.nn
from loss_landscapes.model_interface.model_parameters import ModelParameters


class ModelWrapper(abc.ABC):
    def __init__(self, modules: list):
        self.modules = modules

    def get_modules(self) -> list:
        return self.modules

    def get_module_parameters(self) -> ModelParameters:
        return ModelParameters([p for module in self.modules for p in module.parameters()])

    def get_module_running_stats(self) -> typing.List[dict]:
        running_stats = []
        for module in self.modules:
            states = dict(module.state_dict())
            params = dict(module.named_parameters())
            stats_dict = { name : states[name] for name in set(states) - set(params) }
            running_stats.append(stats_dict)
        return running_stats

    def load_module_running_stats(self, running_stats: typing.List[dict]):
        for i in range(len(self.modules)):
            module = self.modules[i]
            stats_dict = running_stats[i]
            for name, param in module.state_dict().items():
                if name in stats_dict:
                    param.copy_(copy.deepcopy(stats_dict[name]))

    def get_module_parameters_rmbn2(self):
        parameters = []
        to_keep_index = []
        bn_removed_parameters = []
        layers_to_tweak = 0
        for module in self.modules:
            for name, param in module.named_parameters():
                if 'conv' not in name:
                    parameters.append(param)
                    to_keep_index.append(0)
                else:
                    layers_to_tweak += 1
                    to_keep_index.append(1)
                    parameters.append(param)
                    bn_removed_parameters.append(param)
        
        print("layers_to_tweak")
        print(layers_to_tweak)
        return ModelParameters(parameters), to_keep_index, ModelParameters(bn_removed_parameters)

    def train(self, mode=True) -> 'ModelWrapper':
        for module in self.modules:
            module.train(mode)
        return self

    def eval(self) -> 'ModelWrapper':
        return self.train(False)

    def requires_grad_(self, requires_grad=True) -> 'ModelWrapper':
        for module in self.modules:
            for p in module.parameters():
                p.requires_grad = requires_grad
        return self

    def zero_grad(self) -> 'ModelWrapper':
        for module in self.modules:
            for p in module.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
        return self

    def parameters(self):
        return itertools.chain([module.parameters() for module in self.modules])

    def named_parameters(self):
        return itertools.chain([module.named_parameters() for module in self.modules])

    @abc.abstractmethod
    def forward(self, x):
        pass


class SimpleModelWrapper(ModelWrapper):
    def __init__(self, model: torch.nn.Module):
        super().__init__([model])

    def forward(self, x):
        return self.modules[0](x)


class GeneralModelWrapper(ModelWrapper):
    def __init__(self, model, modules: list, forward_fn):
        super().__init__(modules)
        self.model = model
        self.forward_fn = forward_fn

    def forward(self, x):
        return self.forward_fn(self.model, x)


def wrap_model(model):
    if isinstance(model, ModelWrapper):
        return model.requires_grad_(False)
    elif isinstance(model, torch.nn.Module):
        return SimpleModelWrapper(model).requires_grad_(False)
    else:
        raise ValueError('Only models of type torch.nn.modules.module.Module can be passed without a wrapper.')
