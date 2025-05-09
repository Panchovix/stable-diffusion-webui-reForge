import glob
import os
import inspect

import safetensors
import torch

from modules import devices, shared, hashes, errors
from torch.nn.init import normal_, xavier_normal_, xavier_uniform_, kaiming_normal_, kaiming_uniform_, zeros_


class HypernetworkModule(torch.nn.Module):
    activation_dict = {
        "linear": torch.nn.Identity,
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "swish": torch.nn.Hardswish,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
    }
    activation_dict.update({cls_name.lower(): cls_obj for cls_name, cls_obj in inspect.getmembers(torch.nn.modules.activation) if inspect.isclass(cls_obj) and cls_obj.__module__ == 'torch.nn.modules.activation'})

    def __init__(self, dim, state_dict=None, layer_structure=None, activation_func=None, weight_init='Normal',
                 add_layer_norm=False, activate_output=False, dropout_structure=None):
        super().__init__()

        self.multiplier = 1.0

        assert layer_structure is not None, "layer_structure must not be None"
        assert layer_structure[0] == 1, "Multiplier Sequence should start with size 1!"
        assert layer_structure[-1] == 1, "Multiplier Sequence should end with size 1!"

        linears = []
        for i in range(len(layer_structure) - 1):

            # Add a fully-connected layer
            linears.append(torch.nn.Linear(int(dim * layer_structure[i]), int(dim * layer_structure[i+1])))

            # Add an activation func except last layer
            if activation_func == "linear" or activation_func is None or (i >= len(layer_structure) - 2 and not activate_output):
                pass
            elif activation_func in self.activation_dict:
                linears.append(self.activation_dict[activation_func]())
            else:
                raise RuntimeError(f'hypernetwork uses an unsupported activation function: {activation_func}')

            # Add layer normalization
            if add_layer_norm:
                linears.append(torch.nn.LayerNorm(int(dim * layer_structure[i+1])))

            # Everything should be now parsed into dropout structure, and applied here.
            # Since we only have dropouts after layers, dropout structure should start with 0 and end with 0.
            if dropout_structure is not None and dropout_structure[i+1] > 0:
                assert 0 < dropout_structure[i+1] < 1, "Dropout probability should be 0 or float between 0 and 1!"
                linears.append(torch.nn.Dropout(p=dropout_structure[i+1]))
            # Code explanation : [1, 2, 1] -> dropout is missing when last_layer_dropout is false. [1, 2, 2, 1] -> [0, 0.3, 0, 0], when its True, [0, 0.3, 0.3, 0].

        self.linear = torch.nn.Sequential(*linears)

        if state_dict is not None:
            self.fix_old_state_dict(state_dict)
            self.load_state_dict(state_dict)
        else:
            for layer in self.linear:
                if type(layer) == torch.nn.Linear or type(layer) == torch.nn.LayerNorm:
                    w, b = layer.weight.data, layer.bias.data
                    if weight_init == "Normal" or type(layer) == torch.nn.LayerNorm:
                        normal_(w, mean=0.0, std=0.01)
                        normal_(b, mean=0.0, std=0)
                    elif weight_init == 'XavierUniform':
                        xavier_uniform_(w)
                        zeros_(b)
                    elif weight_init == 'XavierNormal':
                        xavier_normal_(w)
                        zeros_(b)
                    elif weight_init == 'KaimingUniform':
                        kaiming_uniform_(w, nonlinearity='leaky_relu' if 'leakyrelu' == activation_func else 'relu')
                        zeros_(b)
                    elif weight_init == 'KaimingNormal':
                        kaiming_normal_(w, nonlinearity='leaky_relu' if 'leakyrelu' == activation_func else 'relu')
                        zeros_(b)
                    else:
                        raise KeyError(f"Key {weight_init} is not defined as initialization!")
        devices.torch_npu_set_device()
        self.to(devices.device)

    def fix_old_state_dict(self, state_dict):
        changes = {
            'linear1.bias': 'linear.0.bias',
            'linear1.weight': 'linear.0.weight',
            'linear2.bias': 'linear.1.bias',
            'linear2.weight': 'linear.1.weight',
        }

        for fr, to in changes.items():
            x = state_dict.get(fr, None)
            if x is None:
                continue

            del state_dict[fr]
            state_dict[to] = x

    def forward(self, x):
        return x + self.linear(x) * self.multiplier


#param layer_structure : sequence used for length, use_dropout : controlling boolean, last_layer_dropout : for compatibility check.
def parse_dropout_structure(layer_structure, use_dropout, last_layer_dropout):
    if layer_structure is None:
        layer_structure = [1, 2, 1]
    if not use_dropout:
        return [0] * len(layer_structure)
    dropout_values = [0]
    dropout_values.extend([0.3] * (len(layer_structure) - 3))
    if last_layer_dropout:
        dropout_values.append(0.3)
    else:
        dropout_values.append(0)
    dropout_values.append(0)
    return dropout_values


class Hypernetwork:
    filename = None
    name = None

    def __init__(self, name=None, enable_sizes=None, layer_structure=None, activation_func=None, weight_init=None, add_layer_norm=False, use_dropout=False, activate_output=False, **kwargs):
        self.filename = None
        self.name = name
        self.layers = {}
        self.step = 0
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.layer_structure = layer_structure
        self.activation_func = activation_func
        self.weight_init = weight_init
        self.add_layer_norm = add_layer_norm
        self.use_dropout = use_dropout
        self.activate_output = activate_output
        self.last_layer_dropout = kwargs.get('last_layer_dropout', True)
        self.dropout_structure = kwargs.get('dropout_structure', None)
        if self.dropout_structure is None:
            self.dropout_structure = parse_dropout_structure(self.layer_structure, self.use_dropout, self.last_layer_dropout)
        self.optimizer_name = None
        self.optimizer_state_dict = None
        self.optional_info = None

        for size in enable_sizes or []:
            self.layers[size] = (
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init,
                                   self.add_layer_norm, self.activate_output, dropout_structure=self.dropout_structure),
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init,
                                   self.add_layer_norm, self.activate_output, dropout_structure=self.dropout_structure),
            )
        self.eval()

    def weights(self):
        res = []
        for layers in self.layers.values():
            for layer in layers:
                res += layer.parameters()
        return res

    def to(self, device):
        for layers in self.layers.values():
            for layer in layers:
                layer.to(device)

        return self

    def set_multiplier(self, multiplier):
        for layers in self.layers.values():
            for layer in layers:
                layer.multiplier = multiplier

        return self

    def eval(self):
        for layers in self.layers.values():
            for layer in layers:
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

    def load(self, filename):
        self.filename = filename
        if self.name is None:
            self.name = os.path.splitext(os.path.basename(filename))[0]

        if filename.endswith(".pt"):
            state_dict = torch.load(filename, map_location='cpu')
        else: # untested
            state_dict = safetensors.torch.load_file(filename, device='cpu')

        self.layer_structure = state_dict.get('layer_structure', [1, 2, 1])
        self.optional_info = state_dict.get('optional_info', None)
        self.activation_func = state_dict.get('activation_func', None)
        self.weight_init = state_dict.get('weight_initialization', 'Normal')
        self.add_layer_norm = state_dict.get('is_layer_norm', False)
        self.dropout_structure = state_dict.get('dropout_structure', None)
        self.use_dropout = True if self.dropout_structure is not None and any(self.dropout_structure) else state_dict.get('use_dropout', False)
        self.activate_output = state_dict.get('activate_output', True)
        self.last_layer_dropout = state_dict.get('last_layer_dropout', False)
        # Dropout structure should have same length as layer structure, Every digits should be in [0,1), and last digit must be 0.
        if self.dropout_structure is None:
            self.dropout_structure = parse_dropout_structure(self.layer_structure, self.use_dropout, self.last_layer_dropout)

        if shared.opts.print_hypernet_extra:
            if self.optional_info is not None:
                print(f"  INFO:\n {self.optional_info}\n")

            print(f"  Layer structure: {self.layer_structure}")
            print(f"  Activation function: {self.activation_func}")
            print(f"  Weight initialization: {self.weight_init}")
            print(f"  Layer norm: {self.add_layer_norm}")
            print(f"  Dropout usage: {self.use_dropout}" )
            print(f"  Activate last layer: {self.activate_output}")
            print(f"  Dropout structure: {self.dropout_structure}")

        optimizer_saved_dict = torch.load(self.filename + '.optim', map_location='cpu') if os.path.exists(self.filename + '.optim') else {}

        if self.shorthash() == optimizer_saved_dict.get('hash', None):
            self.optimizer_state_dict = optimizer_saved_dict.get('optimizer_state_dict', None)
        else:
            self.optimizer_state_dict = None
        if self.optimizer_state_dict:
            self.optimizer_name = optimizer_saved_dict.get('optimizer_name', 'AdamW')
            if shared.opts.print_hypernet_extra:
                print("Loaded existing optimizer from checkpoint")
                print(f"Optimizer name is {self.optimizer_name}")
        else:
            self.optimizer_name = "AdamW"
            if shared.opts.print_hypernet_extra:
                print("No saved optimizer exists in checkpoint")

        for size, sd in state_dict.items():
            if type(size) == int:
                self.layers[size] = (
                    HypernetworkModule(size, sd[0], self.layer_structure, self.activation_func, self.weight_init,
                                       self.add_layer_norm, self.activate_output, self.dropout_structure),
                    HypernetworkModule(size, sd[1], self.layer_structure, self.activation_func, self.weight_init,
                                       self.add_layer_norm, self.activate_output, self.dropout_structure),
                )

        self.name = state_dict.get('name', self.name)
        self.step = state_dict.get('step', 0)
        self.sd_checkpoint = state_dict.get('sd_checkpoint', None)
        self.sd_checkpoint_name = state_dict.get('sd_checkpoint_name', None)
        self.eval()

    def shorthash(self):
        sha256 = hashes.sha256(self.filename, f'hypernet/{self.name}')

        return sha256[0:10] if sha256 else None


def list_hypernetworks(path):
    res = {}

    for filename in sorted(glob.iglob(os.path.join(path, '**/*.pt'), recursive=True), key=str.lower):
        name = os.path.splitext(os.path.basename(filename))[0]
        # Prevent a hypothetical "None.pt" from being listed.
        if name != "None":
            res[name] = filename
    for filename in sorted(glob.iglob(os.path.join(path, '**/*.safetensors'), recursive=True), key=str.lower):
        name = os.path.splitext(os.path.basename(filename))[0]
        if name != "None":
            res[name] = filename
    return res


def load_hypernetwork(name):
    path = shared.hypernetworks.get(name, None)

    if path is None:
        return None

    try:
        hypernetwork = Hypernetwork()
        hypernetwork.load(path)
        return hypernetwork
    except Exception:
        errors.report(f"Error loading hypernetwork {path}", exc_info=True)
        return None


def load_hypernetworks(names, multipliers=None):
    already_loaded = {}

    for hypernetwork in shared.loaded_hypernetworks:
        if hypernetwork.name in names:
            already_loaded[hypernetwork.name] = hypernetwork

    shared.loaded_hypernetworks.clear()

    for i, name in enumerate(names):
        hypernetwork = already_loaded.get(name, None)
        if hypernetwork is None:
            hypernetwork = load_hypernetwork(name)

        if hypernetwork is None:
            continue

        hypernetwork.set_multiplier(multipliers[i] if multipliers else 1.0)
        shared.loaded_hypernetworks.append(hypernetwork)


def stack_conds(conds):
    if len(conds) == 1:
        return torch.stack(conds)

    # same as in reconstruct_multicond_batch
    token_count = max([x.shape[0] for x in conds])
    for i in range(len(conds)):
        if conds[i].shape[0] != token_count:
            last_vector = conds[i][-1:]
            last_vector_repeated = last_vector.repeat([token_count - conds[i].shape[0], 1])
            conds[i] = torch.vstack([conds[i], last_vector_repeated])

    return torch.stack(conds)
