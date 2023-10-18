import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
import torchonn as onn
from torchonn.models import ONNBaseModel
import torch.optim as optim
from pyutils.compute import merge_chunks
from torchonn.op.matrix_parametrization import RealUnitaryDecomposerBatch
import pdb

from torchonn.op.mzi_op import (
    PhaseQuantizer,
    checkerboard_to_vector,
    phase_to_voltage,
    upper_triangle_to_vector,
    vector_to_checkerboard,
    vector_to_upper_triangle,
    voltage_to_phase,
)

class Learner(nn.Module):
    """

    """

    def __init__(self, config, imgc, imgsz, mb):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()


        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        self.actual_vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        self.conv_layers = []
        self.linear_layers = []

        device = torch.device('cuda')

        decompose_alg = 'clements'

        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
                conv_layer = onn.layers.MZIBlockConv2d(
                    in_channels = param[1],
                    out_channels = param[0],
                    kernel_size = param[2],
                    stride = param[4],
                    padding = param[5],
                    dilation=1,
                    bias=True,
                    mode="phase",
                    decompose_alg=decompose_alg,
                    photodetect=True,
                    device=device,
                    miniblock = mb,
                )
                conv_layer.reset_parameters()
                self.conv_layers.append(conv_layer)
                self.vars.append(conv_layer.phase_U)
                self.vars.append(conv_layer.phase_S)
                self.vars.append(conv_layer.phase_V)
                self.vars.append(conv_layer.S_scale)
                self.vars.append(conv_layer.bias)

                self.actual_vars.append(conv_layer.phase_U)
                self.actual_vars.append(conv_layer.phase_V)
                self.actual_vars.append(conv_layer.bias)

            elif name == 'linear':
                linear_layer = onn.layers.MZIBlockLinear(
                    in_features=param[1],
                    out_features=param[0],
                    bias=True,
                    miniblock=mb,
                    mode="phase",
                    decompose_alg=decompose_alg,
                    photodetect=True,
                    device=device,
                )
                linear_layer.reset_parameters()
                self.linear_layers.append(linear_layer)
                self.vars.append(linear_layer.phase_U)
                self.vars.append(linear_layer.phase_S)
                self.vars.append(linear_layer.phase_V)
                self.vars.append(linear_layer.S_scale)
                self.vars.append(linear_layer.bias)

                self.actual_vars.append(linear_layer.phase_U)
                self.actual_vars.append(linear_layer.phase_V)
                self.actual_vars.append(linear_layer.bias)

            elif name == 'bn':
                w = nn.Parameter(torch.ones(param[0]))
                b = nn.Parameter(torch.zeros(param[0]))
                self.vars.append(w)
                self.vars.append(b)

                self.actual_vars.append(w)
                self.actual_vars.append(b)

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['relu','flatten']:
                continue
            else:
                raise NotImplementedError

    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetuning, however, in finetuning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weights.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.actual_vars

        vars_idx = 0
        vars_actual_idx = 0
        conv_idx = 0
        linear_idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name == 'conv2d':
                # just use self.conv_layers[conv_idx] as a way to hold info about what the layer should be shaped as
                conv_layer = self.conv_layers[conv_idx]
                phase_u, phase_s, phase_v, s_scale, b = vars[vars_actual_idx], self.vars[vars_idx + 1], vars[vars_actual_idx + 1], self.vars[vars_idx + 3], vars[vars_actual_idx + 2]
                u = conv_layer.decomposer.reconstruct(conv_layer.delta_list_U, conv_layer.decomposer.v2m(phase_u))
                v = conv_layer.decomposer.reconstruct(conv_layer.delta_list_V, conv_layer.decomposer.v2m(phase_v))
                s = phase_s.cos().mul_(s_scale)
                weight = u.matmul(s.unsqueeze(-1) * v)
                weight = merge_chunks(weight)[: conv_layer.out_channels, : conv_layer.in_channels_flat].view(
                    -1, conv_layer.in_channels, conv_layer.kernel_size[0], conv_layer.kernel_size[1]
                )
                x = F.conv2d(
                    x,
                    weight,
                    bias=b,
                    stride=conv_layer.stride,
                    padding=conv_layer.padding,
                    dilation=conv_layer.dilation,
                    groups=conv_layer.groups,
                )
                x = x + b.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                conv_idx += 1
                vars_idx += 5
                vars_actual_idx += 3
            elif name == 'linear':
                # just use self.linear_layers[linear_idx] as a way to hold info about what the layer should be shaped as
                linear_layer = self.linear_layers[linear_idx]
                phase_u, phase_s, phase_v, s_scale, b = vars[vars_actual_idx], self.vars[vars_idx + 1], vars[vars_actual_idx + 1], self.vars[vars_idx + 3], vars[vars_actual_idx + 2]
                u = linear_layer.decomposer.reconstruct(linear_layer.delta_list_U, linear_layer.decomposer.v2m(phase_u))
                v = linear_layer.decomposer.reconstruct(linear_layer.delta_list_V, linear_layer.decomposer.v2m(phase_v))
                s = phase_s.cos().mul_(s_scale)
                weight = u.matmul(s.unsqueeze(-1) * v)
                weight = merge_chunks(weight)[: linear_layer.out_features, : linear_layer.in_features]
                x = F.linear(x, weight, bias=b)
                x = x + b.unsqueeze(0)
                linear_idx += 1
                vars_idx += 5
                vars_actual_idx += 3
            elif name == 'bn':
                w, b = vars[vars_actual_idx], vars[vars_actual_idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                vars_idx += 2
                vars_actual_idx += 2
                bn_idx += 2

            elif name == 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert bn_idx == len(self.vars_bn)
        return x


    def zero_grad(self, vars=None):
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.actual_vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.actual_vars
