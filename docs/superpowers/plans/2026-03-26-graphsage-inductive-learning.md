# GraphSAGE Inductive Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an inductive embedding pipeline using GraphSAGE so new Stellar accounts can be scored for anomalies without retraining.

**Architecture:** A multi-layer GraphSAGE encoder produces embeddings for arbitrary nodes via K-hop neighbor sampling. An orchestrator ties together graph snapshots, feature computation, and encoding. The existing Deep SVDD model consumes these embeddings for anomaly scoring.

**Tech Stack:** PyTorch, existing `SAGEConv` from `astroml/features/gnn/sage.py`, existing `compute_node_features()` from `astroml/features/node_features.py`, existing `window_snapshot()` from `astroml/features/graph/snapshot.py`, Hydra configs.

**Spec:** `docs/superpowers/specs/2026-03-26-graphsage-inductive-learning-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `astroml/features/gnn/sampler.py` | Create | Multi-hop neighbor sampling |
| `tests/test_multi_hop_sampler.py` | Create | Sampler unit tests |
| `astroml/models/sage_encoder.py` | Create | Multi-layer GraphSAGE encoder |
| `astroml/models/__init__.py` | Modify | Export new encoder |
| `tests/test_sage_encoder.py` | Create | Encoder unit tests |
| `astroml/pipeline/__init__.py` | Create | Pipeline package init |
| `astroml/pipeline/inductive.py` | Create | Embedding orchestrator |
| `tests/test_inductive_pipeline.py` | Create | Pipeline unit tests |
| `astroml/pipeline/scoring.py` | Create | SVDD integration for anomaly scoring |
| `tests/test_inductive_scoring.py` | Create | Scoring integration tests |
| `astroml/training/train_sage.py` | Create | Hydra-driven inductive training script |
| `tests/test_inductive_training.py` | Create | Training loop integration tests |
| `configs/model/sage.yaml` | Create | Encoder hyperparameters |
| `configs/sampling/default.yaml` | Create | Neighbor sampling config |
| `configs/experiment/inductive.yaml` | Create | Inductive experiment preset |
| `tests/test_inductive_generalization.py` | Create | End-to-end inductive validation |

---

### Task 1: MultiHopSampler

**Files:**
- Create: `astroml/features/gnn/sampler.py`
- Test: `tests/test_multi_hop_sampler.py`

- [ ] **Step 1: Write failing test for single-hop sampling**

```python
# tests/test_multi_hop_sampler.py
from __future__ import annotations

import torch
from astroml.features.gnn.sampler import MultiHopSampler


def test_single_hop_sampling():
    """Single-hop with fanout [2] returns one adjacency layer."""
    # Graph: 0->2, 1->2, 3->2, 0->1
    edge_index = torch.tensor([[0, 1, 3, 0], [2, 2, 2, 1]], dtype=torch.long)
    sampler = MultiHopSampler(edge_index=edge_index, num_nodes=4, fanout=[2])

    target = torch.tensor([2], dtype=torch.long)
    adjs, all_nodes = sampler.sample(target)

    assert len(adjs) == 1
    edge_idx, size = adjs[0]
    assert edge_idx.shape[0] == 2  # [2, E]
    # Target node 2 has 3 neighbors (0, 1, 3); fanout=2 so at most 2 sampled
    assert edge_idx.shape[1] <= 2
    # all_nodes includes target + sampled neighbors
    assert 2 in all_nodes.tolist()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_multi_hop_sampler.py::test_single_hop_sampling -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'astroml.features.gnn.sampler'`

- [ ] **Step 3: Implement MultiHopSampler**

```python
# astroml/features/gnn/sampler.py
from __future__ import annotations

from typing import List, Tuple

import torch

from astroml.features.gnn.sage import sample_neighbors


class MultiHopSampler:
    """K-hop neighbor sampler with configurable fanout per layer.

    Parameters
    ----------
    edge_index : Tensor [2, E]
        Full graph edge index.
    num_nodes : int
        Total number of nodes in the graph.
    fanout : list[int]
        Number of neighbors to sample at each hop. Length determines number of hops.
        Example: [25, 10] means 25 neighbors at hop-1, 10 at hop-2.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        fanout: List[int],
    ) -> None:
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.fanout = fanout

    def sample(
        self, target_nodes: torch.Tensor
    ) -> Tuple[List[Tuple[torch.Tensor, Tuple[int, int]]], torch.Tensor]:
        """Sample multi-hop neighborhood around target_nodes.

        Returns
        -------
        adjs : list of (edge_index, (num_src, num_dst))
            One entry per hop, ordered from outermost hop to innermost.
            edge_index uses local indices relative to the sampled subgraph.
        all_nodes : Tensor
            Union of all node IDs touched, ordered so that target_nodes
            appear at indices [0..len(target_nodes)-1].
        """
        adjs: List[Tuple[torch.Tensor, Tuple[int, int]]] = []
        current_nodes = target_nodes.clone()

        node_sets: List[torch.Tensor] = [current_nodes]

        for num_samples in reversed(self.fanout):
            src, dst = sample_neighbors(self.edge_index, current_nodes, num_samples)

            if src.numel() > 0:
                new_nodes = src[~torch.isin(src, current_nodes)]
                if new_nodes.numel() > 0:
                    current_nodes = torch.cat([current_nodes, new_nodes.unique()])
            node_sets.append(current_nodes.clone())

        # Build all_nodes: target first, then rest
        all_nodes = current_nodes
        # Ensure target_nodes are at the front
        non_target = all_nodes[~torch.isin(all_nodes, target_nodes)]
        all_nodes = torch.cat([target_nodes, non_target])

        # Build local-index adjacency lists, outermost hop first
        global_to_local = {int(n): i for i, n in enumerate(all_nodes.tolist())}

        # Re-sample to build adjs with local indices (iterate outermost to innermost)
        layer_dst_nodes = target_nodes
        for num_samples in self.fanout:
            src, dst = sample_neighbors(self.edge_index, layer_dst_nodes, num_samples)

            if src.numel() == 0:
                local_edge = torch.zeros((2, 0), dtype=torch.long)
                num_src = len(all_nodes)
                num_dst = len(layer_dst_nodes)
            else:
                local_src = torch.tensor([global_to_local[int(s)] for s in src.tolist()], dtype=torch.long)
                local_dst = torch.tensor([global_to_local[int(d)] for d in dst.tolist()], dtype=torch.long)
                local_edge = torch.stack([local_src, local_dst])

                # Expand layer_dst for next hop
                new_neighbors = src[~torch.isin(src, layer_dst_nodes)]
                layer_dst_nodes = torch.cat([layer_dst_nodes, new_neighbors.unique()])

                num_src = len(all_nodes)
                num_dst = local_edge[1].max().item() + 1 if local_edge.numel() > 0 else 0

            adjs.append((local_edge, (num_src, int(num_dst))))

        return adjs, all_nodes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_multi_hop_sampler.py::test_single_hop_sampling -v`
Expected: PASS

- [ ] **Step 5: Write failing test for multi-hop sampling**

```python
# Append to tests/test_multi_hop_sampler.py

def test_two_hop_sampling():
    """Two-hop sampling returns two adjacency layers."""
    # Chain: 0->1->2->3
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    sampler = MultiHopSampler(edge_index=edge_index, num_nodes=4, fanout=[2, 2])

    target = torch.tensor([3], dtype=torch.long)
    adjs, all_nodes = sampler.sample(target)

    assert len(adjs) == 2
    # Target node (3) should be first in all_nodes
    assert all_nodes[0].item() == 3
    # Should reach at least node 2 (hop-1) and node 1 (hop-2)
    node_list = all_nodes.tolist()
    assert 2 in node_list
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_multi_hop_sampler.py::test_two_hop_sampling -v`
Expected: PASS (implementation already handles multi-hop)

- [ ] **Step 7: Write failing test for isolated node**

```python
# Append to tests/test_multi_hop_sampler.py

def test_isolated_node():
    """Isolated target node returns empty adjacency lists."""
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    sampler = MultiHopSampler(edge_index=edge_index, num_nodes=3, fanout=[5])

    target = torch.tensor([2], dtype=torch.long)  # Node 2 has no edges
    adjs, all_nodes = sampler.sample(target)

    assert len(adjs) == 1
    edge_idx, size = adjs[0]
    assert edge_idx.shape[1] == 0  # No edges sampled
    assert 2 in all_nodes.tolist()


def test_deterministic_with_seed():
    """Same seed produces same sampling results."""
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [5, 5, 5, 5, 5]], dtype=torch.long)
    sampler = MultiHopSampler(edge_index=edge_index, num_nodes=6, fanout=[2])
    target = torch.tensor([5], dtype=torch.long)

    torch.manual_seed(42)
    adjs1, nodes1 = sampler.sample(target)

    torch.manual_seed(42)
    adjs2, nodes2 = sampler.sample(target)

    assert torch.equal(adjs1[0][0], adjs2[0][0])
    assert torch.equal(nodes1, nodes2)
```

- [ ] **Step 8: Run all sampler tests**

Run: `pytest tests/test_multi_hop_sampler.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add astroml/features/gnn/sampler.py tests/test_multi_hop_sampler.py
git commit -m "feat: add MultiHopSampler for K-hop neighbor sampling (#70)"
```

---

### Task 2: InductiveSAGEEncoder

**Files:**
- Create: `astroml/models/sage_encoder.py`
- Modify: `astroml/models/__init__.py:1-13`
- Test: `tests/test_sage_encoder.py`

- [ ] **Step 1: Write failing test for encoder output shape**

```python
# tests/test_sage_encoder.py
from __future__ import annotations

import torch
from astroml.models.sage_encoder import InductiveSAGEEncoder


def test_encoder_output_shape():
    """Encoder produces correct output dimensions."""
    encoder = InductiveSAGEEncoder(
        input_dim=8, hidden_dim=16, output_dim=8, num_layers=2,
        dropout=0.0, aggregator='mean',
    )
    # 10 total nodes, 3 target nodes
    x = torch.randn(10, 8)
    # Two adjacency layers (one per SAGEConv layer)
    # Layer 0 (outermost): edges among all 10 nodes, targeting first 6
    adj0_edge = torch.tensor([[7, 8, 9, 6], [0, 1, 2, 3]], dtype=torch.long)
    adj0 = (adj0_edge, (10, 6))
    # Layer 1 (innermost): edges among first 6 nodes, targeting first 3
    adj1_edge = torch.tensor([[3, 4, 5], [0, 1, 2]], dtype=torch.long)
    adj1 = (adj1_edge, (6, 3))

    adjs = [adj0, adj1]
    out = encoder(x, adjs)

    assert out.shape == (3, 8)


def test_encoder_single_layer():
    """Single-layer encoder works correctly."""
    encoder = InductiveSAGEEncoder(
        input_dim=4, hidden_dim=4, output_dim=4, num_layers=1,
        dropout=0.0, aggregator='mean',
    )
    x = torch.randn(5, 4)
    adj_edge = torch.tensor([[3, 4], [0, 1]], dtype=torch.long)
    adjs = [(adj_edge, (5, 2))]

    out = encoder(x, adjs)
    assert out.shape == (2, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sage_encoder.py::test_encoder_output_shape -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'astroml.models.sage_encoder'`

- [ ] **Step 3: Implement InductiveSAGEEncoder**

```python
# astroml/models/sage_encoder.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sage_encoder.py -v`
Expected: All PASS

- [ ] **Step 5: Write failing test for gradient flow**

```python
# Append to tests/test_sage_encoder.py

def test_encoder_gradient_flow():
    """Gradients flow through all layers."""
    encoder = InductiveSAGEEncoder(
        input_dim=4, hidden_dim=8, output_dim=4, num_layers=2,
        dropout=0.0, aggregator='mean',
    )
    x = torch.randn(6, 4, requires_grad=True)
    adj0_edge = torch.tensor([[3, 4, 5], [0, 1, 2]], dtype=torch.long)
    adj1_edge = torch.tensor([[1, 2], [0, 0]], dtype=torch.long)
    adjs = [(adj0_edge, (6, 3)), (adj1_edge, (3, 1))]

    out = encoder(x, adjs)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    # All conv layers should have gradients
    for conv in encoder.convs:
        assert conv.lin_l.weight.grad is not None
        assert conv.lin_r.weight.grad is not None
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_sage_encoder.py::test_encoder_gradient_flow -v`
Expected: PASS

- [ ] **Step 7: Update models __init__.py**

Add the new encoder to the package exports in `astroml/models/__init__.py`:

```python
# astroml/models/__init__.py
"""Machine learning models for AstroML."""

from .deep_svdd import DeepSVDD, DeepSVDDNetwork
from .deep_svdd_trainer import DeepSVDDTrainer, FraudDetectionDeepSVDD
from .gcn import GCN
from .sage_encoder import InductiveSAGEEncoder

__all__ = [
    'DeepSVDD',
    'DeepSVDDNetwork',
    'DeepSVDDTrainer',
    'FraudDetectionDeepSVDD',
    'GCN',
    'InductiveSAGEEncoder',
]
```

- [ ] **Step 8: Run all encoder tests**

Run: `pytest tests/test_sage_encoder.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add astroml/models/sage_encoder.py astroml/models/__init__.py tests/test_sage_encoder.py
git commit -m "feat: add InductiveSAGEEncoder multi-layer model (#70)"
```

---

### Task 3: InductiveGraphSAGE Pipeline

**Files:**
- Create: `astroml/pipeline/__init__.py`
- Create: `astroml/pipeline/inductive.py`
- Test: `tests/test_inductive_pipeline.py`

- [ ] **Step 1: Create pipeline package**

```python
# astroml/pipeline/__init__.py
"""Inference pipelines for AstroML."""
```

- [ ] **Step 2: Write failing test for embed_nodes**

```python
# tests/test_inductive_pipeline.py
from __future__ import annotations

import torch
from astroml.pipeline.inductive import InductiveGraphSAGE
from astroml.models.sage_encoder import InductiveSAGEEncoder
from astroml.features.gnn.sampler import MultiHopSampler


def _make_edges():
    """Synthetic edges as dicts for compute_node_features."""
    return [
        {'src': 'A', 'dst': 'B', 'amount': 100.0, 'timestamp': 1000.0, 'asset': 'XLM'},
        {'src': 'B', 'dst': 'C', 'amount': 50.0, 'timestamp': 2000.0, 'asset': 'XLM'},
        {'src': 'C', 'dst': 'D', 'amount': 25.0, 'timestamp': 3000.0, 'asset': 'USD'},
        {'src': 'A', 'dst': 'C', 'amount': 75.0, 'timestamp': 1500.0, 'asset': 'XLM'},
        {'src': 'D', 'dst': 'A', 'amount': 10.0, 'timestamp': 3500.0, 'asset': 'BTC'},
    ]


def test_embed_nodes_returns_target_embeddings():
    """embed_nodes returns embeddings for requested target nodes only."""
    edges = _make_edges()
    encoder = InductiveSAGEEncoder(
        input_dim=8, hidden_dim=16, output_dim=8,
        num_layers=2, dropout=0.0, aggregator='mean',
    )
    pipeline = InductiveGraphSAGE(encoder=encoder, fanout=[3, 2], device='cpu')

    result = pipeline.embed_nodes(edges, target_nodes=['C', 'D'], ref_time=4000.0)

    assert 'C' in result
    assert 'D' in result
    assert 'A' not in result
    assert result['C'].shape == (8,)
    assert result['D'].shape == (8,)


def test_embed_nodes_single_transaction_account():
    """Account with a single transaction still gets a valid embedding."""
    edges = [
        {'src': 'X', 'dst': 'Y', 'amount': 1.0, 'timestamp': 100.0, 'asset': 'XLM'},
    ]
    encoder = InductiveSAGEEncoder(
        input_dim=8, hidden_dim=8, output_dim=4,
        num_layers=1, dropout=0.0, aggregator='mean',
    )
    pipeline = InductiveGraphSAGE(encoder=encoder, fanout=[5], device='cpu')

    result = pipeline.embed_nodes(edges, target_nodes=['Y'], ref_time=200.0)

    assert 'Y' in result
    assert result['Y'].shape == (4,)
    assert torch.isfinite(result['Y']).all()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_inductive_pipeline.py::test_embed_nodes_returns_target_embeddings -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'astroml.pipeline'`

- [ ] **Step 4: Implement InductiveGraphSAGE**

```python
# astroml/pipeline/inductive.py
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from astroml.features.node_features import compute_node_features
from astroml.features.graph.snapshot import window_snapshot
from astroml.features.gnn.sampler import MultiHopSampler
from astroml.models.sage_encoder import InductiveSAGEEncoder


# Column order from compute_node_features output
_FEATURE_COLS = [
    'in_degree', 'out_degree', 'total_received', 'total_sent',
    'account_age', 'first_seen', 'unique_asset_count', 'asset_entropy',
]


class InductiveGraphSAGE:
    """Orchestrator for inductive node embedding via GraphSAGE.

    Ties together graph snapshots, feature computation, neighbor sampling,
    and the multi-layer SAGEConv encoder into a single callable.

    Parameters
    ----------
    encoder : InductiveSAGEEncoder
        Trained (or untrained) multi-layer encoder.
    fanout : list[int]
        Neighbors per hop, passed to MultiHopSampler.
    device : str
        Torch device for computation.
    """

    def __init__(
        self,
        encoder: InductiveSAGEEncoder,
        fanout: List[int],
        device: str = 'cpu',
    ) -> None:
        self.encoder = encoder.to(device)
        self.fanout = fanout
        self.device = device

    def embed_nodes(
        self,
        edges: List[Dict],
        target_nodes: List[str],
        ref_time: float,
    ) -> Dict[str, torch.Tensor]:
        """Compute embeddings for target_nodes given a set of edges.

        Parameters
        ----------
        edges : list of dict
            Transaction edges with keys: src, dst, amount, timestamp, asset.
        target_nodes : list of str
            Account IDs to produce embeddings for.
        ref_time : float
            Reference timestamp for feature computation.

        Returns
        -------
        dict mapping node_id -> Tensor of shape [output_dim]
        """
        # 1. Compute node features
        feat_df = compute_node_features(edges, ref_time=ref_time)
        all_node_ids = list(feat_df.index)
        node_to_idx = {n: i for i, n in enumerate(all_node_ids)}

        # 2. Build edge_index in integer space
        src_indices = []
        dst_indices = []
        for e in edges:
            s, d = e.get('src'), e.get('dst')
            if s in node_to_idx and d in node_to_idx:
                src_indices.append(node_to_idx[s])
                dst_indices.append(node_to_idx[d])

        if not src_indices:
            # No edges: return zero embeddings
            out_dim = self.encoder.convs[-1].out_dim
            return {n: torch.zeros(out_dim) for n in target_nodes}

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)

        # 3. Convert target_nodes to integer indices
        target_idx = torch.tensor(
            [node_to_idx[n] for n in target_nodes if n in node_to_idx],
            dtype=torch.long,
        )
        valid_targets = [n for n in target_nodes if n in node_to_idx]

        if target_idx.numel() == 0:
            out_dim = self.encoder.convs[-1].out_dim
            return {n: torch.zeros(out_dim) for n in target_nodes}

        # 4. Sample neighborhoods
        sampler = MultiHopSampler(edge_index, len(all_node_ids), self.fanout)
        adjs, sampled_nodes = sampler.sample(target_idx)

        # 5. Slice feature matrix to sampled nodes
        feat_matrix = feat_df.loc[
            [all_node_ids[i] for i in sampled_nodes.tolist()]
        ][_FEATURE_COLS].values
        x = torch.tensor(feat_matrix, dtype=torch.float32).to(self.device)

        # 6. Forward through encoder
        self.encoder.eval()
        with torch.no_grad():
            embeddings = self.encoder(x, adjs)

        # 7. Map back to node IDs
        result = {}
        for i, node_id in enumerate(valid_targets):
            result[node_id] = embeddings[i].cpu()

        return result

    def embed_snapshot(
        self,
        edges: Sequence,
        start_ts: int,
        end_ts: int,
        target_nodes: List[str],
        ref_time: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute embeddings using a time-windowed snapshot.

        Parameters
        ----------
        edges : sequence of snapshot Edge objects
            Full edge list (will be filtered by time window).
        start_ts : int
            Start of time window (inclusive).
        end_ts : int
            End of time window (inclusive).
        target_nodes : list of str
            Account IDs to embed.
        ref_time : float, optional
            Reference time for features. Defaults to end_ts.

        Returns
        -------
        dict mapping node_id -> Tensor of shape [output_dim]
        """
        _, window_edges = window_snapshot(edges, start_ts, end_ts)

        # Convert snapshot Edge objects to dicts for compute_node_features
        edge_dicts = [
            {'src': e.src, 'dst': e.dst, 'amount': 0.0, 'timestamp': float(e.timestamp)}
            for e in window_edges
        ]

        return self.embed_nodes(
            edge_dicts,
            target_nodes,
            ref_time=ref_time if ref_time is not None else float(end_ts),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_inductive_pipeline.py -v`
Expected: All PASS

- [ ] **Step 6: Write failing test for embed_snapshot**

```python
# Append to tests/test_inductive_pipeline.py
from astroml.features.graph.snapshot import Edge


def test_embed_snapshot_filters_by_time():
    """embed_snapshot only uses edges within the time window."""
    edges = [
        Edge(src='A', dst='B', timestamp=100),
        Edge(src='B', dst='C', timestamp=200),
        Edge(src='C', dst='D', timestamp=300),
        Edge(src='D', dst='E', timestamp=400),
    ]
    encoder = InductiveSAGEEncoder(
        input_dim=8, hidden_dim=8, output_dim=4,
        num_layers=1, dropout=0.0, aggregator='mean',
    )
    pipeline = InductiveGraphSAGE(encoder=encoder, fanout=[5], device='cpu')

    # Only include edges at timestamps 200-300
    result = pipeline.embed_snapshot(edges, start_ts=200, end_ts=300, target_nodes=['C'])

    assert 'C' in result
    assert result['C'].shape == (4,)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_inductive_pipeline.py::test_embed_snapshot_filters_by_time -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add astroml/pipeline/__init__.py astroml/pipeline/inductive.py tests/test_inductive_pipeline.py
git commit -m "feat: add InductiveGraphSAGE embedding orchestrator (#70)"
```

---

### Task 4: InductiveAnomalyScorer

**Files:**
- Create: `astroml/pipeline/scoring.py`
- Test: `tests/test_inductive_scoring.py`

- [ ] **Step 1: Write failing test for anomaly scoring**

```python
# tests/test_inductive_scoring.py
from __future__ import annotations

import torch
import numpy as np
from astroml.pipeline.scoring import InductiveAnomalyScorer
from astroml.pipeline.inductive import InductiveGraphSAGE
from astroml.models.sage_encoder import InductiveSAGEEncoder
from astroml.models.deep_svdd import DeepSVDD


def _make_edges():
    return [
        {'src': 'A', 'dst': 'B', 'amount': 100.0, 'timestamp': 1000.0, 'asset': 'XLM'},
        {'src': 'B', 'dst': 'C', 'amount': 50.0, 'timestamp': 2000.0, 'asset': 'XLM'},
        {'src': 'C', 'dst': 'D', 'amount': 25.0, 'timestamp': 3000.0, 'asset': 'USD'},
        {'src': 'A', 'dst': 'D', 'amount': 10.0, 'timestamp': 3500.0, 'asset': 'XLM'},
    ]


def test_score_new_accounts():
    """Scorer returns finite anomaly scores for new accounts."""
    encoder = InductiveSAGEEncoder(
        input_dim=8, hidden_dim=16, output_dim=8,
        num_layers=2, dropout=0.0, aggregator='mean',
    )
    pipeline = InductiveGraphSAGE(encoder=encoder, fanout=[3, 2], device='cpu')

    # Create SVDD with matching input_dim = encoder output_dim
    svdd = DeepSVDD(input_dim=8, hidden_dims=[16, 8], dropout=0.0, device='cpu')
    # Manually set center so predict() works without fitting
    svdd.center = torch.zeros(8)

    scorer = InductiveAnomalyScorer(pipeline, svdd)
    scores = scorer.score_new_accounts(_make_edges(), ['C', 'D'], ref_time=4000.0)

    assert 'C' in scores
    assert 'D' in scores
    assert np.isfinite(scores['C'])
    assert np.isfinite(scores['D'])


def test_score_concatenated_mode():
    """Concatenated mode produces scores using embeddings + raw features."""
    encoder = InductiveSAGEEncoder(
        input_dim=8, hidden_dim=8, output_dim=4,
        num_layers=1, dropout=0.0, aggregator='mean',
    )
    pipeline = InductiveGraphSAGE(encoder=encoder, fanout=[3], device='cpu')

    # input_dim = encoder output (4) + raw features (8) = 12
    svdd = DeepSVDD(input_dim=12, hidden_dims=[8, 4], dropout=0.0, device='cpu')
    svdd.center = torch.zeros(4)

    scorer = InductiveAnomalyScorer(pipeline, svdd, mode='concatenated')
    scores = scorer.score_new_accounts(_make_edges(), ['B'], ref_time=4000.0)

    assert 'B' in scores
    assert np.isfinite(scores['B'])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inductive_scoring.py::test_score_new_accounts -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'astroml.pipeline.scoring'`

- [ ] **Step 3: Implement InductiveAnomalyScorer**

```python
# astroml/pipeline/scoring.py
from __future__ import annotations

from typing import Dict, List

import torch

from astroml.pipeline.inductive import InductiveGraphSAGE, _FEATURE_COLS
from astroml.features.node_features import compute_node_features
from astroml.models.deep_svdd import DeepSVDD


class InductiveAnomalyScorer:
    """Connects inductive GraphSAGE embeddings to Deep SVDD for anomaly scoring.

    Parameters
    ----------
    pipeline : InductiveGraphSAGE
        Embedding pipeline for producing node representations.
    svdd : DeepSVDD
        Trained Deep SVDD model for anomaly scoring.
    mode : str
        'embeddings_only' feeds GraphSAGE embeddings to SVDD directly.
        'concatenated' concatenates embeddings with raw node features.
    """

    def __init__(
        self,
        pipeline: InductiveGraphSAGE,
        svdd: DeepSVDD,
        mode: str = 'embeddings_only',
    ) -> None:
        assert mode in ('embeddings_only', 'concatenated')
        self.pipeline = pipeline
        self.svdd = svdd
        self.mode = mode

    def score_new_accounts(
        self,
        edges: List[Dict],
        account_ids: List[str],
        ref_time: float,
    ) -> Dict[str, float]:
        """Produce anomaly scores for the given accounts.

        Parameters
        ----------
        edges : list of dict
            Transaction edges for graph context.
        account_ids : list of str
            Accounts to score.
        ref_time : float
            Reference timestamp for feature computation.

        Returns
        -------
        dict mapping account_id -> anomaly score (float).
            Higher scores indicate more anomalous.
        """
        embeddings = self.pipeline.embed_nodes(edges, account_ids, ref_time)

        if self.mode == 'concatenated':
            feat_df = compute_node_features(edges, ref_time=ref_time)

        results: Dict[str, float] = {}
        for node_id in account_ids:
            if node_id not in embeddings:
                results[node_id] = float('inf')
                continue

            emb = embeddings[node_id]

            if self.mode == 'concatenated':
                if node_id in feat_df.index:
                    raw = torch.tensor(
                        feat_df.loc[node_id][_FEATURE_COLS].values,
                        dtype=torch.float32,
                    )
                else:
                    raw = torch.zeros(len(_FEATURE_COLS))
                svdd_input = torch.cat([emb, raw]).unsqueeze(0)
            else:
                svdd_input = emb.unsqueeze(0)

            score = self.svdd.predict(svdd_input)
            results[node_id] = float(score[0])

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_inductive_scoring.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add astroml/pipeline/scoring.py tests/test_inductive_scoring.py
git commit -m "feat: add InductiveAnomalyScorer for SVDD integration (#70)"
```

---

### Task 5: Hydra Configuration

**Files:**
- Create: `configs/model/sage.yaml`
- Create: `configs/sampling/default.yaml`
- Create: `configs/experiment/inductive.yaml`

- [ ] **Step 1: Create sage model config**

```yaml
# configs/model/sage.yaml
_target_: astroml.models.sage_encoder.InductiveSAGEEncoder

input_dim: 8
hidden_dim: 64
output_dim: 32
num_layers: 2
dropout: 0.5
aggregator: "mean"
```

- [ ] **Step 2: Create sampling config**

```yaml
# configs/sampling/default.yaml
fanout: [25, 10]
batch_size: 512
```

- [ ] **Step 3: Create inductive experiment config**

```yaml
# configs/experiment/inductive.yaml
defaults:
  - /model: sage
  - /training: default

sampling:
  fanout: [25, 10]
  batch_size: 512

experiment:
  name: "inductive_sage"
  temporal_split_ratio: 0.7
  loss: "reconstruction"
  svdd_mode: "embeddings_only"
```

- [ ] **Step 4: Verify configs parse correctly**

Run: `python -c "from omegaconf import OmegaConf; c = OmegaConf.load('configs/model/sage.yaml'); print(c)"`
Expected: Prints the config dict without errors.

Run: `python -c "from omegaconf import OmegaConf; c = OmegaConf.load('configs/sampling/default.yaml'); print(c)"`
Expected: Prints the config dict without errors.

- [ ] **Step 5: Commit**

```bash
git add configs/model/sage.yaml configs/sampling/default.yaml configs/experiment/inductive.yaml
git commit -m "feat: add Hydra configs for inductive GraphSAGE (#70)"
```

---

### Task 6: Inductive Training Script

**Files:**
- Create: `astroml/training/train_sage.py`
- Test: `tests/test_inductive_training.py`

- [ ] **Step 1: Write failing test for training loop**

```python
# tests/test_inductive_training.py
from __future__ import annotations

import torch
from astroml.training.train_sage import train_epoch, build_reconstruction_target
from astroml.models.sage_encoder import InductiveSAGEEncoder
from astroml.features.gnn.sampler import MultiHopSampler


def _make_synthetic_graph(num_nodes=20, num_edges=40):
    """Create a random graph with features."""
    edge_index = torch.stack([
        torch.randint(0, num_nodes, (num_edges,)),
        torch.randint(0, num_nodes, (num_edges,)),
    ])
    features = torch.randn(num_nodes, 8)
    return edge_index, features


def test_train_epoch_loss_decreases():
    """Training for a few epochs should decrease loss."""
    torch.manual_seed(42)
    edge_index, features = _make_synthetic_graph(20, 40)

    encoder = InductiveSAGEEncoder(
        input_dim=8, hidden_dim=16, output_dim=8,
        num_layers=2, dropout=0.0, aggregator='mean',
    )
    sampler = MultiHopSampler(edge_index, num_nodes=20, fanout=[5, 3])
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    train_nodes = torch.arange(20)

    loss1 = train_epoch(encoder, sampler, features, edge_index, train_nodes, optimizer, batch_size=10)
    # Run a few more epochs
    for _ in range(9):
        loss_n = train_epoch(encoder, sampler, features, edge_index, train_nodes, optimizer, batch_size=10)

    assert loss_n < loss1


def test_build_reconstruction_target():
    """Reconstruction target is mean of neighbor raw features."""
    edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]], dtype=torch.long)
    features = torch.tensor([
        [1.0, 0.0],  # node 0
        [0.0, 1.0],  # node 1
        [1.0, 1.0],  # node 2
        [0.0, 0.0],  # node 3 (target)
    ])
    target_nodes = torch.tensor([3])

    targets = build_reconstruction_target(edge_index, features, target_nodes)

    # Mean of nodes 0, 1, 2 features = [2/3, 2/3]
    expected = torch.tensor([[2.0 / 3.0, 2.0 / 3.0]])
    assert torch.allclose(targets, expected, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inductive_training.py::test_train_epoch_loss_decreases -v`
Expected: FAIL with `ImportError: cannot import name 'train_epoch' from 'astroml.training.train_sage'`

- [ ] **Step 3: Implement training script**

```python
# astroml/training/train_sage.py
from __future__ import annotations

"""Inductive GraphSAGE training with unsupervised reconstruction loss.

The encoder learns to produce embeddings that can reconstruct the mean
of each node's immediate neighbors' raw features. This produces general-
purpose embeddings without requiring labels.
"""

import logging

import torch
from torch import nn

from astroml.models.sage_encoder import InductiveSAGEEncoder
from astroml.features.gnn.sampler import MultiHopSampler

logger = logging.getLogger(__name__)


def build_reconstruction_target(
    edge_index: torch.Tensor,
    features: torch.Tensor,
    target_nodes: torch.Tensor,
) -> torch.Tensor:
    """Compute mean neighbor features for each target node.

    Parameters
    ----------
    edge_index : Tensor [2, E]
        Full graph edge index.
    features : Tensor [N, F]
        Raw node features.
    target_nodes : Tensor [B]
        Nodes to compute targets for.

    Returns
    -------
    Tensor [B, F] - mean of neighbors' features per target node.
    """
    src, dst = edge_index
    result = torch.zeros(target_nodes.size(0), features.size(1))

    for i, node in enumerate(target_nodes):
        neighbors = src[dst == node]
        if neighbors.numel() > 0:
            result[i] = features[neighbors].mean(dim=0)

    return result


def train_epoch(
    encoder: InductiveSAGEEncoder,
    sampler: MultiHopSampler,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    train_nodes: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 512,
    device: str = 'cpu',
) -> float:
    """Train encoder for one epoch using mini-batch reconstruction loss.

    Parameters
    ----------
    encoder : InductiveSAGEEncoder
    sampler : MultiHopSampler
    features : Tensor [N, F] - raw node features
    edge_index : Tensor [2, E] - full graph edges (for reconstruction targets)
    train_nodes : Tensor [T] - node indices to train on
    optimizer : torch.optim.Optimizer
    batch_size : int
    device : str

    Returns
    -------
    float - average loss for the epoch.
    """
    encoder.train()
    decoder = nn.Linear(encoder.convs[-1].out_dim, features.size(1)).to(device)
    decoder_opt = torch.optim.Adam(decoder.parameters(), lr=optimizer.defaults['lr'])

    perm = torch.randperm(train_nodes.size(0))
    total_loss = 0.0
    num_batches = 0

    for start in range(0, train_nodes.size(0), batch_size):
        batch_indices = perm[start:start + batch_size]
        batch_nodes = train_nodes[batch_indices]

        # Sample neighborhoods
        adjs, sampled_node_ids = sampler.sample(batch_nodes)

        # Slice features for sampled nodes
        x = features[sampled_node_ids].to(device)

        # Forward
        embeddings = encoder(x, adjs)

        # Reconstruction target: mean of neighbors' raw features
        targets = build_reconstruction_target(edge_index, features, batch_nodes).to(device)

        # Decode and compute loss
        decoded = decoder(embeddings)
        loss = nn.functional.mse_loss(decoded, targets)

        optimizer.zero_grad()
        decoder_opt.zero_grad()
        loss.backward()
        optimizer.step()
        decoder_opt.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_inductive_training.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add astroml/training/train_sage.py tests/test_inductive_training.py
git commit -m "feat: add inductive GraphSAGE training script with reconstruction loss (#70)"
```

---

### Task 7: Inductive Generalization Test

**Files:**
- Test: `tests/test_inductive_generalization.py`

- [ ] **Step 1: Write the generalization test**

This is the core validation that issue #70 demands: train on one subgraph, embed nodes from a completely disjoint subgraph.

```python
# tests/test_inductive_generalization.py
from __future__ import annotations

"""End-to-end test: train GraphSAGE on subgraph A, produce embeddings for
unseen nodes in subgraph B. Verifies the core inductive learning property."""

import torch
from astroml.models.sage_encoder import InductiveSAGEEncoder
from astroml.features.gnn.sampler import MultiHopSampler
from astroml.training.train_sage import train_epoch


def test_inductive_generalization():
    """Embeddings for unseen nodes are non-zero and vary across nodes."""
    torch.manual_seed(123)

    # --- Subgraph A (training): nodes 0-9, dense ---
    num_train_nodes = 10
    train_edges = []
    for i in range(num_train_nodes):
        for j in range(num_train_nodes):
            if i != j and torch.rand(1).item() > 0.5:
                train_edges.append([i, j])
    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t()
    train_features = torch.randn(num_train_nodes, 8)

    # --- Train encoder ---
    encoder = InductiveSAGEEncoder(
        input_dim=8, hidden_dim=16, output_dim=8,
        num_layers=2, dropout=0.0, aggregator='mean',
    )
    sampler = MultiHopSampler(train_edge_index, num_train_nodes, fanout=[5, 3])
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    train_nodes = torch.arange(num_train_nodes)

    for _ in range(20):
        train_epoch(encoder, sampler, train_features, train_edge_index,
                    train_nodes, optimizer, batch_size=5)

    # --- Subgraph B (unseen): nodes 0-4, completely disjoint accounts ---
    num_test_nodes = 5
    test_edges = []
    for i in range(num_test_nodes):
        for j in range(num_test_nodes):
            if i != j and torch.rand(1).item() > 0.5:
                test_edges.append([i, j])
    if not test_edges:
        test_edges = [[0, 1], [1, 0]]  # ensure at least one edge
    test_edge_index = torch.tensor(test_edges, dtype=torch.long).t()
    test_features = torch.randn(num_test_nodes, 8)

    # --- Embed unseen nodes using the trained encoder ---
    test_sampler = MultiHopSampler(test_edge_index, num_test_nodes, fanout=[5, 3])
    target_nodes = torch.arange(num_test_nodes)

    encoder.eval()
    with torch.no_grad():
        adjs, sampled = test_sampler.sample(target_nodes)
        x = test_features[sampled]
        embeddings = encoder(x, adjs)

    # Verify: embeddings are non-zero
    assert embeddings.shape == (num_test_nodes, 8)
    assert (embeddings.abs().sum(dim=1) > 0).all(), "All embeddings should be non-zero"

    # Verify: embeddings vary across nodes (not all identical)
    pairwise_diffs = torch.cdist(embeddings, embeddings)
    # At least some pairs should differ
    off_diagonal = pairwise_diffs[~torch.eye(num_test_nodes, dtype=torch.bool)]
    assert (off_diagonal > 1e-6).any(), "Embeddings should vary across nodes"

    # Verify: embeddings are finite
    assert torch.isfinite(embeddings).all()
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_inductive_generalization.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_inductive_generalization.py
git commit -m "test: add end-to-end inductive generalization validation (#70)"
```

---

## Self-Review Checklist

**Spec coverage:**
- InductiveSAGEEncoder: Task 2
- MultiHopSampler: Task 1
- InductiveGraphSAGE: Task 3
- InductiveAnomalyScorer: Task 4
- Training script: Task 6
- Hydra configs: Task 5
- Testing (unit + integration + generalization): Tasks 1-4, 6, 7
- Both batch and on-demand inference: Task 3 (`embed_nodes` for on-demand, `embed_snapshot` for batch)
- Evolution path to Approach B: documented in spec, no code needed in this plan

**Placeholder scan:** No TBD/TODO/placeholder patterns found.

**Type consistency:** `adjs` format (List[Tuple[Tensor, Tuple[int, int]]]) is consistent across MultiHopSampler.sample(), InductiveSAGEEncoder.forward(), and all tests. `_FEATURE_COLS` is defined in inductive.py and imported in scoring.py. Method names match across all tasks.
