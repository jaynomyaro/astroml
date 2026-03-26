from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from astroml.features.gnn.sage import SAGEConv


class InductiveSAGEEncoder(nn.Module):
    """Multi-layer GraphSAGE encoder for inductive node embedding.

    Stacks SAGEConv layers with ReLU and dropout. Designed to consume
    adjacency lists from MultiHopSampler, processing from outermost hop
    to innermost (target nodes).

    Parameters
    ----------
    input_dim : int
        Dimension of input node features (8 for compute_node_features output).
    hidden_dim : int
        Hidden layer dimension.
    output_dim : int
        Output embedding dimension.
    num_layers : int
        Number of SAGEConv layers. Must match len(adjs) passed to forward().
    dropout : float
        Dropout probability between layers.
    aggregator : str
        Aggregation strategy for SAGEConv ('mean' or 'gcn').
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggregator: str = 'mean',
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(SAGEConv(input_dim, output_dim, aggregator=aggregator))
        else:
            self.convs.append(SAGEConv(input_dim, hidden_dim, aggregator=aggregator))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggregator=aggregator))
            self.convs.append(SAGEConv(hidden_dim, output_dim, aggregator=aggregator))

    def forward(
        self,
        x: torch.Tensor,
        adjs: List[Tuple[torch.Tensor, Tuple[int, int]]],
    ) -> torch.Tensor:
        """Forward pass through stacked SAGEConv layers.

        Parameters
        ----------
        x : Tensor [N_total, input_dim]
            Node features for all sampled nodes.
        adjs : list of (edge_index, (num_src, num_dst))
            One per layer, outermost hop first. Each edge_index uses local
            indices. num_dst defines how many destination (target) nodes
            for that layer.

        Returns
        -------
        Tensor [N_target, output_dim]
            Embeddings for the innermost target nodes.
        """
        for i, (edge_index, (num_src, num_dst)) in enumerate(adjs):
            x_dst = x[:num_dst]
            x_pair = (x, x_dst)

            x = self.convs[i](x_pair, edge_index)

            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
