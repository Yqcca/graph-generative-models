from collections.abc import Sequence
import functools
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_max

from torchdrug import core, layers
from torchdrug.layers import MessagePassingBase

class GATV2Conv(MessagePassingBase):
    """
    Graph Attention V2 Network proposed in 'How Attentive are Graph Attention Networks?'.
    Graph Attention V2 Network: https://arxiv.org/abs/2105.14491

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        num_head (int, optional): number of attention heads
        negative_slope (float, optional): negative slope of leaky relu activation
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    eps = 1e-10

    def __init__(self, input_dim, output_dim, edge_input_dim=None, num_head=1, negative_slope=0.2, concat=True,
                 batch_norm=False, activation="relu"):
        super(GATV2Conv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.num_head = num_head
        self.concat = concat
        self.leaky_relu = functools.partial(F.leaky_relu, negative_slope=negative_slope)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if output_dim % num_head != 0:
            raise ValueError("Expect output_dim to be a multiplier of num_head, but found `%d` and `%d`"
                             % (output_dim, num_head))

        self.linear = nn.Linear(input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, output_dim)
        else:
            self.edge_linear = None
        self.query = nn.Parameter(torch.zeros(num_head, output_dim * 2 // num_head))
        nn.init.kaiming_uniform_(self.query, negative_slope, mode="fan_in")

    def message(self, graph, input):
        # add self loop
        node_in = torch.cat([graph.edge_list[:, 0], torch.arange(graph.num_node, device=graph.device)])
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1)
        hidden = self.linear(input)

        key = hidden[torch.stack([node_in, node_out], dim=-1)]
        if self.edge_linear:
            edge_input = self.edge_linear(graph.edge_feature.float())
            edge_input = torch.cat([edge_input, torch.zeros(graph.num_node, self.output_dim, device=graph.device)])
            key += edge_input.unsqueeze(-1).reshape(key.size()[0], 1, key.size()[2])

        key = key.view(-1, *self.query.shape)
        weight = torch.einsum("hd, nhd -> nh", self.query, self.leaky_relu(key))

        weight = weight - scatter_max(weight, node_out, dim=0, dim_size=graph.num_node)[0][node_out]
        attention = weight.exp() * edge_weight
        # why mean? because with mean we have normalized message scale across different node degrees
        normalizer = scatter_mean(attention, node_out, dim=0, dim_size=graph.num_node)[node_out]
        attention = attention / (normalizer + self.eps)
        value = hidden[node_in].view(-1, self.num_head, self.query.shape[-1] // 2)
        attention = attention.unsqueeze(-1).expand_as(value)
        message = (attention * value).flatten(1)
        return message

    def aggregate(self, graph, message):
        # add self loop
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        update = scatter_mean(message, node_out, dim=0, dim_size=graph.num_node)
        return update

    def combine(self, input, update):
        output = update
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
    

class GATV2(nn.Module, core.Configurable):
    """
    Graph Attention V2 Network proposed in 'How Attentive are Graph Attention Networks?'.
    Graph Attention V2 Network: https://arxiv.org/abs/2105.14491

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_relation (int): number of relation
        edge_input_dim (int, optional): dimension of edge features
        num_head (int, optional): number of attention heads
        negative_slope (float, optional): negative slope of leaky relu activation
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, hidden_dims, num_relation=None, edge_input_dim=None, num_head=1, negative_slope=0.2, short_cut=False,
                 batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(GATV2, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.num_relation = num_relation
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(GATV2Conv(self.dims[i], self.dims[i + 1], edge_input_dim, num_head,
                                                         negative_slope, batch_norm, activation))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

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
        layer_input = input

        for layer in self.layers:
            hidden = layer(graph, layer_input)
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
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }