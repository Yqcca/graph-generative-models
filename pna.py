from typing import Optional, Dict, List

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils import checkpoint
from torch_geometric.utils import degree
from torch_scatter import scatter
from torchdrug import layers, core
from torchdrug.layers import MessagePassingBase


# PNA aggregators

def aggregate_sum(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='sum')


def aggregate_mean(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='mean')


def aggregate_min(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='min')


def aggregate_max(src: Tensor, index: Tensor, dim_size: Optional[int]):
    return scatter(src, index, 0, None, dim_size, reduce='max')


def aggregate_var(src, index, dim_size):
    mean = aggregate_mean(src, index, dim_size)
    mean_squares = aggregate_mean(src * src, index, dim_size)
    return mean_squares - mean * mean


def aggregate_std(src, index, dim_size):
    return torch.sqrt(torch.relu(aggregate_var(src, index, dim_size)) + 1e-5)


# PNA scalers

def scale_identity(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src


def scale_amplification(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src * (torch.log(deg + 1) / avg_deg['log'])


def scale_attenuation(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    scale = avg_deg['log'] / torch.log(deg + 1)
    scale[deg == 0] = 1
    return src * scale


def scale_linear(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    return src * (deg / avg_deg['linear'])


def scale_inverse_linear(src: Tensor, deg: Tensor, avg_deg: Dict[str, float]):
    scale = avg_deg['linear'] / deg
    scale[deg == 0] = 1
    return src * scale


AGGREGATORS = {
    'sum': aggregate_sum,
    'mean': aggregate_mean,
    'min': aggregate_min,
    'max': aggregate_max,
    'var': aggregate_var,
    'std': aggregate_std,
}

SCALERS = {
    'identity': scale_identity,
    'amplification': scale_amplification,
    'attenuation': scale_attenuation,
    'linear': scale_linear,
    'inverse_linear': scale_inverse_linear
}


class PNALayer(MessagePassingBase):
    """
    The Principal Neighbourhood Aggregation graph convolution operator from
    `Principal Neighbourhood Aggregation for Graph Nets`_.

    .. _Principal Neighbourhood Aggregation for Graph Nets:
        https://arxiv.org/pdf/2004.05718.pdf

    Parameters:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        aggregators (list of str): Set of aggregation function identifiers,
            namely "sum", "mean", "min", "max", "var" and "std".
        scalers: (list of str): Set of scaling function identifiers, namely
            "identity", "amplification", "attenuation", "linear" and "inverse_linear".
        deg (Tensor): Histogram of in-degrees of nodes in the training set, used by scalers to normalise.
        edge_dim (int, optional): Edge feature dimensionality (in case there are any).
        towers (int, optional): Number of towers.
        pre_layers (int, optional): Number of transformation layers before aggregation.
        post_layers (int, optional): Number of transformation layers after aggregation.
        divide_input (bool, optional): Whether the input features should be split between towers or not.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 aggregators: List[str], scalers: List[str], deg: Tensor,
                 edge_dim: Optional[int] = None, towers: int = 1,
                 pre_layers: int = 1, post_layers: int = 1,
                 divide_input: bool = False):
        super(PNALayer, self).__init__()

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scaler] for scaler in scalers]
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        deg = deg.to(torch.float)
        total_no_vertices = deg.sum()
        bin_degrees = torch.arange(len(deg))
        self.avg_deg: Dict[str, float] = {
            'linear': ((bin_degrees * deg).sum() / total_no_vertices).item(),
            'log': (((bin_degrees + 1).log() * deg).sum() / total_no_vertices).item(),
            'exp': ((bin_degrees.exp() * deg).sum() / total_no_vertices).item(),
        }

        if self.edge_dim is not None:
            self.edge_encoder = nn.Linear(edge_dim, self.F_in)

        self.pre_mlps = nn.ModuleList()
        self.post_mlps = nn.ModuleList()
        for _ in range(towers):
            pre_modules = [nn.Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                pre_modules += [nn.ReLU()]
                pre_modules += [nn.Linear(self.F_in, self.F_in)]
            self.pre_mlps.append(nn.Sequential(*pre_modules))

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            post_modules = [nn.Linear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                post_modules += [nn.ReLU()]
                post_modules += [nn.Linear(self.F_out, self.F_out)]
            self.post_mlps.append(nn.Sequential(*post_modules))

        self.update_mlp = nn.Linear(out_channels, out_channels)

    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        node_out = graph.edge_list[:, 1]
        if graph.num_edge:
            edge_attr = self.edge_encoder(graph.edge_feature.float())
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            h = torch.cat([input[node_in], input[node_out], edge_attr], dim=-1)
            message = [mlp(h[:, i]) for i, mlp in enumerate(self.pre_mlps)]
        else:
            message = [torch.zeros(0, self.F_in, device=graph.device) for _ in range(self.towers)]
        return torch.stack(message, dim=1)

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.view(-1, 1, 1)
        edge_weight = edge_weight.repeat(1, self.towers, 1)
        update = [aggr(message * edge_weight, node_out, graph.num_node) for aggr in self.aggregators]
        update = torch.cat(update, dim=-1)
        deg = degree(node_out, graph.num_node, dtype=message.dtype).view(-1, 1, 1)
        update = [scaler(update, deg, self.avg_deg) for scaler in self.scalers]
        return torch.cat(update, dim=-1)

    def combine(self, input, update):
        output = torch.cat([input, update], dim=-1)
        output = [mlp(output[:, i]) for i, mlp in enumerate(self.post_mlps)]
        output = torch.cat(output, dim=1)
        return self.update_mlp(output)

    def forward(self, graph, input):
        if self.divide_input:
            input = input.view(-1, self.towers, self.F_in)
        else:
            input = input.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        if self.gradient_checkpoint:
            update = checkpoint.checkpoint(self._message_and_aggregate, *graph.to_tensors(), input)
        else:
            update = self.message_and_aggregate(graph, input)
        output = self.combine(input, update)
        return output

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, towers={self.towers}, dim={self.dim})')


class PNA(nn.Module, core.Configurable):
    """
    Graph Substructure Network proposed in `Improving Graph Neural Network Expressivity
    via Subgraph Isomorphism Counting`_.

    This implements the GSN-v (vertex-count) variant in the original paper.

    .. _Improving Graph Neural Network Expressivity via Subgraph Isomorphism Counting:
        https://arxiv.org/pdf/2006.09252.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dim (int): hidden dimension
        edge_input_dim (int): dimension of edge features
        num_relation (int): number of relations
        num_layer (int): number of hidden layers
        aggregators (list of str): set of aggregation function identifiers,
            namely "sum", "mean", "min", "max", "var" and "std"
        scalers: (list of str): set of scaling function identifiers, namely
            "identity", "amplification", "attenuation", "linear" and "inverse_linear"
        deg (Tensor): histogram of in-degrees of nodes in the training set, used by scalers to normalise
        num_tower (int, optional): number of towers
        num_pre_layer (int, optional): number of MLP layers in each pre-transformation network
        num_post_layer (int, optional): number of MLP layers in each post-transformation network
        divide_input (bool, optional): whether the input features should be split between towers or not
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, hidden_dim, edge_input_dim, num_relation, num_layer, aggregators, scalers, deg,
                 num_tower=1, num_pre_layer=1, num_post_layer=1, divide_input=False, short_cut=False, batch_norm=False,
                 activation='relu', concat_hidden=False, readout='sum'):
        super(PNA, self).__init__()

        self.input_dim = input_dim
        self.edge_input_dim = edge_input_dim
        if concat_hidden:
            feature_dim = hidden_dim * num_layer
        else:
            feature_dim = hidden_dim
        self.output_dim = feature_dim
        self.num_relation = num_relation
        self.num_layer = num_layer
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.node_encoder = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(
                PNALayer(in_channels=hidden_dim, out_channels=hidden_dim, aggregators=aggregators, scalers=scalers,
                         deg=deg, edge_dim=edge_input_dim, towers=num_tower, pre_layers=num_pre_layer,
                         post_layers=num_post_layer, divide_input=divide_input))

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if readout == 'sum':
            self.readout = layers.SumReadout()
        elif readout == 'mean':
            self.readout = layers.MeanReadout()
        else:
            raise ValueError(f'Unknown readout {readout}')

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).
        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = self.node_encoder(input)

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.batch_norm:
                hidden = self.batch_norm(hidden)
            if self.activation:
                hidden = self.activation(hidden)

            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        graph_feature = self.readout(graph, node_feature)

        return {
            'graph_feature': graph_feature,
            'node_feature': node_feature
        }
