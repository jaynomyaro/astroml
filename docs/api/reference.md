# API Reference Documentation

## Overview

This is a comprehensive API reference for all classes, functions, and methods available in the AstroML framework. Each entry includes detailed parameter descriptions, return types, and usage examples.

## Table of Contents

1. [Ingestion Module](#ingestion-module)
2. [Graph Module](#graph-module)
3. [Models Module](#models-module)
4. [Features Module](#features-module)
5. [Training Module](#training-module)
6. [Validation Module](#validation-module)
7. [Configuration Module](#configuration-module)
8. [Utility Functions](#utility-functions)

## Ingestion Module

### IngestionService

```python
class IngestionService:
    def __init__(self, state_store: Optional[StateStore] = None) -> None
    def ingest(
        self,
        start_ledger: Optional[int] = None,
        end_ledger: Optional[int] = None,
        fetch_fn: Optional[Callable[[int], object]] = None,
        process_fn: Optional[Callable[[int, object], None]] = None,
    ) -> IngestionResult
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `state_store` | `Optional[StateStore]` | No | `None` | State management instance |
| `start_ledger` | `Optional[int]` | No | `None` | Starting ledger ID (inclusive) |
| `end_ledger` | `Optional[int]` | No | `None` | Ending ledger ID (inclusive) |
| `fetch_fn` | `Optional[Callable[[int], object]]` | No | `None` | Function to fetch ledger data |
| `process_fn` | `Optional[Callable[[int, object], None]]` | No | `None` | Function to process ledger data |

#### Returns

| Type | Description |
|------|-------------|
| `IngestionResult` | Container with ingestion statistics |

#### Example

```python
service = IngestionService()
result = service.ingest(
    start_ledger=1000000,
    end_ledger=1000100,
    fetch_fn=lambda ledger_id: fetch_stellar_ledger(ledger_id),
    process_fn=lambda ledger_id, data: store_ledger_data(ledger_id, data)
)
print(f"Processed: {len(result.processed)} ledgers")
```

### StateStore

```python
class StateStore:
    def load(self) -> IngestionState
    def save(self, state: IngestionState) -> None
    def mark_processed(self, ledger_id: int) -> None
    def get_last_processed_ledger(self) -> Optional[int]
```

#### Methods

##### load()

Load the current ingestion state.

**Returns:** `IngestionState` - Current state object

##### save()

Save the ingestion state.

**Parameters:**
- `state` (IngestionState): State to save

##### mark_processed()

Mark a ledger as processed.

**Parameters:**
- `ledger_id` (int): Ledger ID to mark

##### get_last_processed_ledger()

Get the last processed ledger ID.

**Returns:** `Optional[int]` - Last processed ledger ID

### EnhancedIngestionService

```python
class EnhancedIngestionService:
    def __init__(
        self,
        state_store: Optional[StateStore] = None,
        config: Optional[IngestionConfig] = None
    ) -> None
    
    async def stream_ingest(
        self,
        start_ledger: int,
        batch_size: int = 100,
        max_concurrent: int = 10
    ) -> AsyncIterator[IngestionResult]
    
    def backfill(
        self,
        start_ledger: int,
        end_ledger: int,
        batch_size: int = 1000,
        parallel_workers: int = 4
    ) -> IngestionResult
```

#### Methods

##### stream_ingest()

Stream ingestion with async processing.

**Parameters:**
- `start_ledger` (int): Starting ledger ID
- `batch_size` (int): Batch size for processing
- `max_concurrent` (int): Maximum concurrent tasks

**Returns:** `AsyncIterator[IngestionResult]` - Stream of results

##### backfill()

Bulk historical data ingestion.

**Parameters:**
- `start_ledger` (int): Starting ledger ID
- `end_ledger` (int): Ending ledger ID
- `batch_size` (int): Batch size
- `parallel_workers` (int): Number of parallel workers

**Returns:** `IngestionResult` - Combined results

### HorizonStream

```python
class HorizonStream:
    def __init__(
        self,
        horizon_url: str,
        network: str = "testnet",
        cursor: Optional[str] = None
    ) -> None
    
    async def stream_transactions(
        self,
        processors: List[Callable[[Transaction], None]]
    ) -> AsyncIterator[Transaction]
    
    async def stream_ledgers(
        self,
        processors: List[Callable[[Ledger], None]]
    ) -> AsyncIterator[Ledger]
```

#### Methods

##### stream_transactions()

Stream real-time transactions.

**Parameters:**
- `processors` (List[Callable]): Processing functions

**Returns:** `AsyncIterator[Transaction]` - Stream of transactions

##### stream_ledgers()

Stream real-time ledgers.

**Parameters:**
- `processors` (List[Callable]): Processing functions

**Returns:** `AsyncIterator[Ledger]` - Stream of ledgers

## Graph Module

### GraphBuilder

```python
class GraphBuilder:
    def __init__(self, config: Optional[GraphConfig] = None) -> None
    
    def build_snapshot(
        self,
        window_days: int,
        min_transactions: int = 1,
        include_temporal: bool = True,
        weight_by_amount: bool = False,
        edge_weights: Optional[Dict[str, float]] = None
    ) -> TemporalGraph
    
    def add_temporal_edges(self, graph: TemporalGraph) -> TemporalGraph
    
    def compute_features(self, graph: TemporalGraph) -> Dict[str, torch.Tensor]
```

#### Methods

##### build_snapshot()

Build a time-window graph snapshot.

**Parameters:**
- `window_days` (int): Time window in days
- `min_transactions` (int): Minimum transactions per node
- `include_temporal` (bool): Include temporal information
- `weight_by_amount` (bool): Weight edges by amount
- `edge_weights` (Optional[Dict]): Custom edge weights

**Returns:** `TemporalGraph` - Constructed graph

##### add_temporal_edges()

Add temporal edges to graph.

**Parameters:**
- `graph` (TemporalGraph): Graph to enhance

**Returns:** `TemporalGraph` - Enhanced graph

##### compute_features()

Compute graph features.

**Parameters:**
- `graph` (TemporalGraph): Input graph

**Returns:** `Dict[str, torch.Tensor]` - Computed features

### TemporalGraph

```python
class TemporalGraph:
    def __init__(self, nodes: List[Node], edges: List[Edge]) -> None
    
    def add_node(self, node: Node) -> None
    def add_edge(self, edge: Edge) -> None
    def get_subgraph(self, time_window: Tuple[datetime, datetime]) -> TemporalGraph
    def degree(self, node_id: str) -> int
    def clustering_coefficient(self, node_id: str) -> float
    def to_networkx(self) -> nx.Graph
    def to_pytorch_geometric(self) -> Data
```

#### Methods

##### add_node()

Add a node to the graph.

**Parameters:**
- `node` (Node): Node to add

##### add_edge()

Add an edge to the graph.

**Parameters:**
- `edge` (Edge): Edge to add

##### get_subgraph()

Get subgraph for time window.

**Parameters:**
- `time_window` (Tuple[datetime, datetime]): Time range

**Returns:** `TemporalGraph` - Subgraph

##### degree()

Get node degree.

**Parameters:**
- `node_id` (str): Node identifier

**Returns:** `int` - Degree of node

##### clustering_coefficient()

Get clustering coefficient.

**Parameters:**
- `node_id` (str): Node identifier

**Returns:** `float` - Clustering coefficient

##### to_networkx()

Convert to NetworkX graph.

**Returns:** `nx.Graph` - NetworkX graph

##### to_pytorch_geometric()

Convert to PyTorch Geometric Data.

**Returns:** `Data` - PyTorch Geometric data object

## Models Module

### GCN

```python
class GCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.5
    ) -> None
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_dim` | `int` | Yes | - | Input feature dimension |
| `hidden_dims` | `List[int]` | Yes | - | Hidden layer dimensions |
| `output_dim` | `int` | Yes | - | Output dimension |
| `dropout` | `float` | No | `0.5` | Dropout rate |

#### Methods

##### forward()

Forward pass through GCN.

**Parameters:**
- `x` (torch.Tensor): Node features [num_nodes, input_dim]
- `edge_index` (torch.Tensor): Edge indices [2, num_edges]

**Returns:** `torch.Tensor` - Log probabilities [num_nodes, output_dim]

### TemporalGCN

```python
class TemporalGCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        temporal_dim: int = 32,
        dropout: float = 0.5
    ) -> None
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time: torch.Tensor,
        node_time: torch.Tensor
    ) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_dim` | `int` | Yes | - | Input feature dimension |
| `hidden_dims` | `List[int]` | Yes | - | Hidden layer dimensions |
| `output_dim` | `int` | Yes | - | Output dimension |
| `temporal_dim` | `int` | No | `32` | Temporal embedding dimension |
| `dropout` | `float` | No | `0.5` | Dropout rate |

#### Methods

##### forward()

Forward pass with temporal information.

**Parameters:**
- `x` (torch.Tensor): Node features
- `edge_index` (torch.Tensor): Edge indices
- `edge_time` (torch.Tensor): Edge timestamps
- `node_time` (torch.Tensor): Node timestamps

**Returns:** `torch.Tensor` - Output predictions

### AnomalyDetector

```python
class AnomalyDetector:
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        threshold: float = 0.95,
        method: str = "autoencoder"
    ) -> None
    
    def fit(self, graph: TemporalGraph, labels: Optional[torch.Tensor] = None) -> None
    def detect(
        self,
        graph: TemporalGraph,
        threshold: Optional[float] = None
    ) -> List[AnomalyResult]
    def predict(self, graph: TemporalGraph) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | `Optional[nn.Module]` | No | `None` | Pre-trained model |
| `threshold` | `float` | No | `0.95` | Detection threshold |
| `method` | `str` | No | `"autoencoder"` | Detection method |

#### Methods

##### fit()

Train the anomaly detector.

**Parameters:**
- `graph` (TemporalGraph): Training graph
- `labels` (Optional[torch.Tensor]): Ground truth labels

##### detect()

Detect anomalies.

**Parameters:**
- `graph` (TemporalGraph): Graph to analyze
- `threshold` (Optional[float]): Detection threshold

**Returns:** `List[AnomalyResult]` - Detected anomalies

##### predict()

Get anomaly scores.

**Parameters:**
- `graph` (TemporalGraph): Graph to analyze

**Returns:** `torch.Tensor` - Anomaly scores

### GraphSAGE

```python
class GraphSAGE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5
    ) -> None
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_dim` | `int` | Yes | - | Input feature dimension |
| `hidden_dims` | `List[int]` | Yes | - | Hidden layer dimensions |
| `output_dim` | `int` | Yes | - | Output dimension |
| `num_layers` | `int` | No | `2` | Number of layers |
| `dropout` | `float` | No | `0.5` | Dropout rate |

#### Methods

##### forward()

Forward pass through GraphSAGE.

**Parameters:**
- `x` (torch.Tensor): Node features
- `edge_index` (torch.Tensor): Edge indices
- `batch` (Optional[torch.Tensor]): Batch indices

**Returns:** `torch.Tensor` - Output predictions

### GAT

```python
class GAT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.5
    ) -> None
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_dim` | `int` | Yes | - | Input feature dimension |
| `hidden_dims` | `List[int]` | Yes | - | Hidden layer dimensions |
| `output_dim` | `int` | Yes | - | Output dimension |
| `num_heads` | `int` | No | `8` | Number of attention heads |
| `dropout` | `float` | No | `0.5` | Dropout rate |

#### Methods

##### forward()

Forward pass through GAT.

**Parameters:**
- `x` (torch.Tensor): Node features
- `edge_index` (torch.Tensor): Edge indices

**Returns:** `torch.Tensor` - Output predictions

## Features Module

### AccountFeatures

```python
class AccountFeatures:
    def __init__(self, config: Optional[FeatureConfig] = None) -> None
    
    def extract_all_features(self, graph: TemporalGraph) -> torch.Tensor
    def extract_degree_features(self, graph: TemporalGraph) -> torch.Tensor
    def extract_transaction_features(self, graph: TemporalGraph) -> torch.Tensor
    def extract_temporal_features(self, graph: TemporalGraph) -> torch.Tensor
    def extract_graph_features(self, graph: TemporalGraph) -> torch.Tensor
```

#### Methods

##### extract_all_features()

Extract all available features.

**Parameters:**
- `graph` (TemporalGraph): Input graph

**Returns:** `torch.Tensor` - Feature matrix [num_nodes, num_features]

##### extract_degree_features()

Extract degree-based features.

**Parameters:**
- `graph` (TemporalGraph): Input graph

**Returns:** `torch.Tensor` - Degree features

##### extract_transaction_features()

Extract transaction-based features.

**Parameters:**
- `graph` (TemporalGraph): Input graph

**Returns:** `torch.Tensor` - Transaction features

##### extract_temporal_features()

Extract temporal features.

**Parameters:**
- `graph` (TemporalGraph): Input graph

**Returns:** `torch.Tensor` - Temporal features

##### extract_graph_features()

Extract graph-level features.

**Parameters:**
- `graph` (TemporalGraph): Input graph

**Returns:** `torch.Tensor` - Graph features

### TransactionFeatures

```python
class TransactionFeatures:
    def __init__(self, config: Optional[FeatureConfig] = None) -> None
    
    def extract_features(self, transactions: List[Transaction]) -> torch.Tensor
    def compute_amount_statistics(self, transactions: List[Transaction]) -> Dict[str, float]
    def compute_temporal_patterns(self, transactions: List[Transaction]) -> Dict[str, float]
```

#### Methods

##### extract_features()

Extract transaction features.

**Parameters:**
- `transactions` (List[Transaction]): Transaction list

**Returns:** `torch.Tensor` - Feature matrix

##### compute_amount_statistics()

Compute amount statistics.

**Parameters:**
- `transactions` (List[Transaction]): Transaction list

**Returns:** `Dict[str, float]` - Statistics

##### compute_temporal_patterns()

Compute temporal patterns.

**Parameters:**
- `transactions` (List[Transaction]): Transaction list

**Returns:** `Dict[str, float]` - Temporal patterns

## Training Module

### Trainer

```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "auto"
    ) -> None
    
    def train(
        self,
        graph: TemporalGraph,
        epochs: int,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> TrainingMetrics
    
    def evaluate(self, graph: TemporalGraph) -> EvaluationMetrics
    def predict(self, graph: TemporalGraph) -> torch.Tensor
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | `nn.Module` | Yes | - | Model to train |
| `optimizer` | `Optional[torch.optim.Optimizer]` | No | `None` | Optimizer |
| `criterion` | `Optional[nn.Module]` | No | `None` | Loss function |
| `device` | `str` | No | `"auto"` | Device for training |

#### Methods

##### train()

Train the model.

**Parameters:**
- `graph` (TemporalGraph): Training graph
- `epochs` (int): Number of epochs
- `learning_rate` (float): Learning rate
- `batch_size` (int): Batch size
- `validation_split` (float): Validation split ratio

**Returns:** `TrainingMetrics` - Training statistics

##### evaluate()

Evaluate the model.

**Parameters:**
- `graph` (TemporalGraph): Test graph

**Returns:** `EvaluationMetrics` - Evaluation statistics

##### predict()

Make predictions.

**Parameters:**
- `graph` (TemporalGraph): Input graph

**Returns:** `torch.Tensor` - Predictions

### Experiment

```python
class Experiment:
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        seed: int = 42
    ) -> None
    
    def run(
        self,
        train_graph: TemporalGraph,
        test_graph: TemporalGraph
    ) -> ExperimentResult
    
    def save_results(self, results: ExperimentResult) -> None
    def load_results(self) -> ExperimentResult
```

#### Methods

##### run()

Run experiment.

**Parameters:**
- `train_graph` (TemporalGraph): Training data
- `test_graph` (TemporalGraph): Test data

**Returns:** `ExperimentResult` - Experiment results

##### save_results()

Save experiment results.

**Parameters:**
- `results` (ExperimentResult): Results to save

##### load_results()

Load experiment results.

**Returns:** `ExperimentResult` - Loaded results

## Validation Module

### Validator

```python
class Validator:
    def __init__(self, metrics: List[str]) -> None
    
    def validate(
        self,
        model: nn.Module,
        test_graph: TemporalGraph,
        test_labels: torch.Tensor
    ) -> ValidationResults
    
    def cross_validate(
        self,
        model: nn.Module,
        graphs: List[TemporalGraph],
        k_folds: int = 5
    ) -> CrossValidationResults
```

#### Methods

##### validate()

Validate model performance.

**Parameters:**
- `model` (nn.Module): Model to validate
- `test_graph` (TemporalGraph): Test graph
- `test_labels` (torch.Tensor): Test labels

**Returns:** `ValidationResults` - Validation results

##### cross_validate()

Perform cross-validation.

**Parameters:**
- `model` (nn.Module): Model to validate
- `graphs` (List[TemporalGraph]): Graph list
- `k_folds` (int): Number of folds

**Returns:** `CrossValidationResults` - Cross-validation results

### MetricsCalculator

```python
class MetricsCalculator:
    def __init__(self) -> None
    
    def calculate_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> float
    def calculate_precision(self, predictions: torch.Tensor, labels: torch.Tensor) -> float
    def calculate_recall(self, predictions: torch.Tensor, labels: torch.Tensor) -> float
    def calculate_f1_score(self, predictions: torch.Tensor, labels: torch.Tensor) -> float
    def calculate_auc_roc(self, predictions: torch.Tensor, labels: torch.Tensor) -> float
    def calculate_confusion_matrix(self, predictions: torch.Tensor, labels: torch.Tensor) -> np.ndarray
```

#### Methods

##### calculate_accuracy()

Calculate accuracy.

**Parameters:**
- `predictions` (torch.Tensor): Model predictions
- `labels` (torch.Tensor): Ground truth labels

**Returns:** `float` - Accuracy score

##### calculate_precision()

Calculate precision.

**Parameters:**
- `predictions` (torch.Tensor): Model predictions
- `labels` (torch.Tensor): Ground truth labels

**Returns:** `float` - Precision score

##### calculate_recall()

Calculate recall.

**Parameters:**
- `predictions` (torch.Tensor): Model predictions
- `labels` (torch.Tensor): Ground truth labels

**Returns:** `float` - Recall score

##### calculate_f1_score()

Calculate F1 score.

**Parameters:**
- `predictions` (torch.Tensor): Model predictions
- `labels` (torch.Tensor): Ground truth labels

**Returns:** `float` - F1 score

##### calculate_auc_roc()

Calculate AUC-ROC.

**Parameters:**
- `predictions` (torch.Tensor): Model predictions
- `labels` (torch.Tensor): Ground truth labels

**Returns:** `float` - AUC-ROC score

##### calculate_confusion_matrix()

Calculate confusion matrix.

**Parameters:**
- `predictions` (torch.Tensor): Model predictions
- `labels` (torch.Tensor): Ground truth labels

**Returns:** `np.ndarray` - Confusion matrix

## Configuration Module

### ConfigLoader

```python
class ConfigLoader:
    def __init__(self) -> None
    
    def load_config(
        self,
        base_files: List[str],
        env_file: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]
    
    def validate_config(self, config: Dict[str, Any]) -> bool
    def get_database_config(self, config: Dict[str, Any]) -> DatabaseConfig
    def get_stellar_config(self, config: Dict[str, Any]) -> StellarConfig
```

#### Methods

##### load_config()

Load configuration from files.

**Parameters:**
- `base_files` (List[str]): Base configuration files
- `env_file` (Optional[str]): Environment-specific file
- `overrides` (Optional[Dict]): Configuration overrides

**Returns:** `Dict[str, Any]` - Loaded configuration

##### validate_config()

Validate configuration.

**Parameters:**
- `config` (Dict[str, Any]): Configuration to validate

**Returns:** `bool` - Validation result

##### get_database_config()

Get database configuration.

**Parameters:**
- `config` (Dict[str, Any]): Full configuration

**Returns:** `DatabaseConfig` - Database configuration

##### get_stellar_config()

Get Stellar configuration.

**Parameters:**
- `config` (Dict[str, Any]): Full configuration

**Returns:** `StellarConfig` - Stellar configuration

## Utility Functions

### Data Processing

```python
def normalize_ledger_data(ledger: RawLedger) -> NormalizedLedger
def normalize_transaction_data(tx: RawTransaction) -> NormalizedTransaction
def validate_account_id(account_id: str) -> bool
def parse_asset_string(asset_string: str) -> Asset
```

#### normalize_ledger_data()

Normalize raw ledger data.

**Parameters:**
- `ledger` (RawLedger): Raw ledger from Stellar

**Returns:** `NormalizedLedger` - Normalized ledger

#### normalize_transaction_data()

Normalize raw transaction data.

**Parameters:**
- `tx` (RawTransaction): Raw transaction from Stellar

**Returns:** `NormalizedTransaction` - Normalized transaction

#### validate_account_id()

Validate Stellar account ID.

**Parameters:**
- `account_id` (str): Account ID to validate

**Returns:** `bool` - Validity result

#### parse_asset_string()

Parse asset string.

**Parameters:**
- `asset_string` (str): Asset string (e.g., "XLM" or "USDC:GA...")

**Returns:** `Asset` - Parsed asset

### Graph Utilities

```python
def compute_graph_metrics(graph: TemporalGraph) -> Dict[str, float]
def find_communities(graph: TemporalGraph, method: str = "louvain") -> Dict[str, int]
def calculate_shortest_paths(graph: TemporalGraph) -> Dict[str, Dict[str, float]]
def detect_cliques(graph: TempGraph, min_size: int = 3) -> List[List[str]]
```

#### compute_graph_metrics()

Compute graph metrics.

**Parameters:**
- `graph` (TemporalGraph): Input graph

**Returns:** `Dict[str, float]` - Graph metrics

#### find_communities()

Find graph communities.

**Parameters:**
- `graph` (TemporalGraph): Input graph
- `method` (str): Community detection method

**Returns:** `Dict[str, int]` - Community assignments

#### calculate_shortest_paths()

Calculate shortest paths.

**Parameters:**
- `graph` (TemporalGraph): Input graph

**Returns:** `Dict[str, Dict[str, float]]` - Shortest paths

#### detect_cliques()

Detect cliques in graph.

**Parameters:**
- `graph` (TemporalGraph): Input graph
- `min_size` (int): Minimum clique size

**Returns:** `List[List[str]]` - Detected cliques

### Model Utilities

```python
def save_model(model: nn.Module, path: str, config: Optional[Dict] = None) -> None
def load_model(path: str) -> Tuple[nn.Module, Dict]
def count_parameters(model: nn.Module) -> int
def freeze_model_parameters(model: nn.Module) -> None
def unfreeze_model_parameters(model: nn.Module) -> None
```

#### save_model()

Save model to disk.

**Parameters:**
- `model` (nn.Module): Model to save
- `path` (str): Save path
- `config` (Optional[Dict]): Model configuration

#### load_model()

Load model from disk.

**Parameters:**
- `path` (str): Model file path

**Returns:** `Tuple[nn.Module, Dict]` - Model and configuration

#### count_parameters()

Count model parameters.

**Parameters:**
- `model` (nn.Module): Model to analyze

**Returns:** `int` - Parameter count

#### freeze_model_parameters()

Freeze model parameters.

**Parameters:**
- `model` (nn.Module): Model to freeze

#### unfreeze_model_parameters()

Unfreeze model parameters.

**Parameters:**
- `model` (nn.Module): Model to unfreeze

### Time Utilities

```python
def parse_stellar_timestamp(timestamp: str) -> datetime
def format_ledger_sequence(sequence: int) -> str
def calculate_time_difference(timestamp1: datetime, timestamp2: datetime) -> float
def get_time_window(start_time: datetime, window_days: int) -> Tuple[datetime, datetime]
```

#### parse_stellar_timestamp()

Parse Stellar timestamp.

**Parameters:**
- `timestamp` (str): Stellar timestamp string

**Returns:** `datetime` - Parsed datetime

#### format_ledger_sequence()

Format ledger sequence.

**Parameters:**
- `sequence` (int): Ledger sequence number

**Returns:** `str` - Formatted sequence

#### calculate_time_difference()

Calculate time difference.

**Parameters:**
- `timestamp1` (datetime): First timestamp
- `timestamp2` (datetime): Second timestamp

**Returns:** `float` - Time difference in seconds

#### get_time_window()

Get time window.

**Parameters:**
- `start_time` (datetime): Start time
- `window_days` (int): Window size in days

**Returns:** `Tuple[datetime, datetime]` - Time window

## Error Classes

### Ingestion Errors

```python
class IngestionError(Exception):
    """Base ingestion error."""
    pass

class LedgerNotFoundError(IngestionError):
    """Ledger not found error."""
    def __init__(self, ledger_id: int):
        self.ledger_id = ledger_id
        super().__init__(f"Ledger {ledger_id} not found")

class StreamError(IngestionError):
    """Streaming error."""
    pass
```

### Graph Errors

```python
class GraphError(Exception):
    """Base graph error."""
    pass

class GraphConstructionError(GraphError):
    """Graph construction error."""
    pass

class FeatureExtractionError(GraphError):
    """Feature extraction error."""
    pass
```

### Model Errors

```python
class ModelError(Exception):
    """Base model error."""
    pass

class TrainingError(ModelError):
    """Training error."""
    pass

class InferenceError(ModelError):
    """Inference error."""
    pass

class ConfigurationError(ModelError):
    """Configuration error."""
    def __init__(self, config_key: str, message: str):
        self.config_key = config_key
        super().__init__(f"Invalid configuration for {config_key}: {message}")
```

## Data Structures

### Node

```python
@dataclass
class Node:
    id: str
    features: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Edge

```python
@dataclass
class Edge:
    source: str
    target: str
    weight: float
    timestamp: datetime
    attributes: Dict[str, Any] = field(default_factory=dict)
```

### Transaction

```python
@dataclass
class Transaction:
    hash: str
    source_account: str
    target_account: str
    amount: float
    asset: Asset
    timestamp: datetime
    memo: Optional[str] = None
    operation_type: str
```

### Asset

```python
@dataclass
class Asset:
    code: str
    issuer: Optional[str]
    asset_type: str  # "native" or "credit_alphanum4"/"credit_alphanum12"
```

### Ledger

```python
@dataclass
class Ledger:
    sequence: int
    hash: str
    timestamp: datetime
    transaction_count: int
    operation_count: int
    successful_transaction_count: int
```

---

This comprehensive API reference provides detailed documentation for all classes, methods, and functions available in the AstroML framework. Each entry includes parameter descriptions, return types, and usage examples to help developers effectively use the framework.
