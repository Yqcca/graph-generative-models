import networkx as nx
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add
from torchdrug import layers, core
from torchdrug.layers import MessagePassingBase


MIN_CYCLE = 3


def generate_node_structural_feature(graph, max_cycle=8):
    if graph.num_node == 0:
        with graph.node():
            graph.node_structural_feature = torch.zeros((0, max_cycle - MIN_CYCLE + 1), device=graph.device)
        return

    graph_nx = nx.Graph(graph.edge_list[:, :2].tolist())
    cycles = [cycle for cycle in nx.cycle_basis(graph_nx) if len(cycle) <= max_cycle]
    cycle_lens = torch.tensor([len(cycle) for cycle in cycles], dtype=torch.int64, device=graph.device) - MIN_CYCLE
    node_in_cycles = torch.tensor([[i in cycle for cycle in cycles] for i in range(graph.num_node)],
                                  dtype=torch.int64, device=graph.device)
    node_cycle_counts = scatter_add(node_in_cycles, cycle_lens, dim_size=max_cycle - MIN_CYCLE + 1)
    with graph.node():
        graph.node_structural_feature = node_cycle_counts


def prepare_GSN_dataset(dataset, max_cycle=8):
    for data in dataset:
        graph = data['graph']
        generate_node_structural_feature(graph, max_cycle)


class GSNLayer(MessagePassingBase):
    """
    Message passing layer of the GSN from `Improving Graph Neural Network Expressivity
    via Subgraph Isomorphism Counting`_.

    This implements the GSN-v (vertex-count) variant in the original paper.

    .. _Improving Graph Neural Network Expressivity via Subgraph Isomorphism Counting:
        https://arxiv.org/pdf/2006.09252.pdf

    Parameters:
        input_dim (int): input dimension
        edge_input_dim (int): dimension of edge features
        max_cycle (int, optional): maximum size of graph substructures
        mlp_hidden_dims (list of int, optional): hidden dims of edge network and update network
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, edge_input_dim, max_cycle=8, mlp_hidden_dims=None,
                 batch_norm=False, activation='relu'):
        super(GSNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.edge_input_dim = edge_input_dim
        self.node_counts_dim = max_cycle - MIN_CYCLE + 1
        if mlp_hidden_dims is None:
            mlp_hidden_dims = []

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(input_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.msg_mlp = layers.MLP(2 * input_dim + 2 * self.node_counts_dim + edge_input_dim,
                                  list(mlp_hidden_dims) + [input_dim], activation)
        self.update_mlp = layers.MLP(2 * input_dim, list(mlp_hidden_dims) + [input_dim], activation)

    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        node_out = graph.edge_list[:, 1]
        if graph.num_edge:
            message = torch.cat([input[node_in], input[node_out],
                                 graph.node_structural_feature[node_in].float(),
                                 graph.node_structural_feature[node_out].float(),
                                 graph.edge_feature.float()], dim=-1)
            message = self.msg_mlp(message)
        else:
            message = torch.zeros(0, self.input_dim, device=graph.device)
        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        return update

    def combine(self, input, update):
        output = torch.cat([input, update], dim=-1)
        output = self.update_mlp(output)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class GSN(nn.Module, core.Configurable):
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
        num_mlp_layer (int, optional): number of MLP layers in each message passing layer
        max_cycle (int, optional): maximum size of graph substructures
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, hidden_dim, edge_input_dim, num_relation, num_layer, num_mlp_layer=2, max_cycle=8,
                 short_cut=False, batch_norm=False, activation='relu', concat_hidden=False, readout='sum'):
        super(GSN, self).__init__()

        self.input_dim = input_dim
        self.edge_input_dim = edge_input_dim
        if concat_hidden:
            feature_dim = hidden_dim * num_layer
        else:
            feature_dim = hidden_dim
        self.output_dim = feature_dim
        self.num_relation = num_relation
        self.num_layer = num_layer
        self.max_cycle = max_cycle
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.linear = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(GSNLayer(hidden_dim, edge_input_dim, max_cycle, [hidden_dim] * (num_mlp_layer - 1),
                                        batch_norm, activation))

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
        if not hasattr(graph, 'node_structural_feature'):
            generate_node_structural_feature(graph, self.max_cycle)

        hiddens = []
        layer_input = self.linear(input)

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
            'graph_feature': graph_feature,
            'node_feature': node_feature
        }
