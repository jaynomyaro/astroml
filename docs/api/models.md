# Machine Learning Models API Documentation

## Overview

The models module provides implementations of graph neural networks and other machine learning models specifically designed for Stellar blockchain data. It includes graph convolutional networks, temporal GNNs, and anomaly detection models.

## Core Models

### GCN (Graph Convolutional Network)

A configurable graph convolutional network for node classification on blockchain transaction graphs.

#### Class Definition

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

#### Constructor Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_dim` | `int` | Yes | - | Dimension of input node features |
| `hidden_dims` | `List[int]` | Yes | - | List of hidden layer dimensions |
| `output_dim` | `int` | Yes | - | Dimension of output (number of classes) |
| `dropout` | `float` | No | `0.5` | Dropout rate for regularization |

#### Methods

##### forward()

Forward pass through the GCN network.

**Parameters:**
- `x` (torch.Tensor): Node feature matrix of shape (num_nodes, input_dim)
- `edge_index` (torch.Tensor): Edge connectivity matrix of shape (2, num_edges)

**Returns:** torch.Tensor - Log probabilities of shape (num_nodes, output_dim)

**Example:**
```python
import torch
from astroml.models import GCN

# Initialize model
model = GCN(
    input_dim=64,
    hidden_dims=[128, 64],
    output_dim=2,  # Binary classification
    dropout=0.5
)

# Forward pass
x = torch.randn(100, 64)  # 100 nodes, 64 features
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  # 3 edges

logits = model(x, edge_index)
probabilities = torch.exp(logits)
```

### TemporalGCN

Time-aware graph convolutional network for temporal graph analysis.

#### Class Definition

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

#### Constructor Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_dim` | `int` | Yes | - | Dimension of input node features |
| `hidden_dims` | `List[int]` | Yes | - | List of hidden layer dimensions |
| `output_dim` | `int` | Yes | - | Dimension of output |
| `temporal_dim` | `int` | No | `32` | Dimension of temporal embeddings |
| `dropout` | `float` | No | `0.5` | Dropout rate |

#### Methods

##### forward()

Forward pass with temporal information.

**Parameters:**
- `x` (torch.Tensor): Node feature matrix
- `edge_index` (torch.Tensor): Edge connectivity matrix
- `edge_time` (torch.Tensor): Timestamps for edges
- `node_time` (torch.Tensor): Timestamps for nodes

**Returns:** torch.Tensor - Output predictions

**Example:**
```python
from astroml.models import TemporalGCN

model = TemporalGCN(
    input_dim=64,
    hidden_dims=[128, 64],
    output_dim=2,
    temporal_dim=32
)

# Temporal forward pass
x = torch.randn(100, 64)
edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
edge_time = torch.tensor([1000, 1005], dtype=torch.float32)
node_time = torch.tensor([990, 1000, 1010], dtype=torch.float32)

output = model(x, edge_index, edge_time, node_time)
```

### AnomalyDetector

Anomaly detection model for identifying suspicious blockchain activity.

#### Class Definition

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

#### Constructor Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | `Optional[nn.Module]` | No | `None` | Pre-trained model for anomaly detection |
| `threshold` | `float` | No | `0.95` | Anomaly detection threshold |
| `method` | `str` | No | `"autoencoder"` | Detection method (autoencoder, isolation_forest, etc.) |

#### Methods

##### fit()

Train the anomaly detection model.

**Parameters:**
- `graph` (TemporalGraph): Training graph data
- `labels` (Optional[torch.Tensor]): Ground truth labels (if available)

##### detect()

Detect anomalies in the graph.

**Parameters:**
- `graph` (TemporalGraph): Graph to analyze
- `threshold` (Optional[float]): Detection threshold override

**Returns:** List[AnomalyResult] - List of detected anomalies

##### predict()

Get anomaly scores for all nodes.

**Parameters:**
- `graph` (TemporalGraph): Graph to analyze

**Returns:** torch.Tensor - Anomaly scores for each node

**Example:**
```python
from astroml.models import AnomalyDetector

detector = AnomalyDetector(method="autoencoder", threshold=0.95)

# Train the detector
detector.fit(training_graph)

# Detect anomalies
anomalies = detector.detect(test_graph)

for anomaly in anomalies:
    print(f"Anomaly detected at node {anomaly.node_id}")
    print(f"Score: {anomaly.score:.4f}")
    print(f"Reason: {anomaly.reason}")
```

### AnomalyResult

Result object for anomaly detection.

#### Class Definition

```python
@dataclass
class AnomalyResult:
    node_id: str
    score: float
    reason: str
    timestamp: datetime
    features: Dict[str, float]
```

## Advanced Models

### GraphSAGE

Graph Sample and Aggregate model for inductive learning on graphs.

#### Class Definition

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

#### Example Usage

```python
from astroml.models import GraphSAGE

model = GraphSAGE(
    input_dim=64,
    hidden_dims=[128, 64],
    output_dim=2,
    num_layers=2
)

# Forward pass with batch processing
x = torch.randn(100, 64)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # 2 graphs

output = model(x, edge_index, batch)
```

### GAT (Graph Attention Network)

Graph Attention Network with attention mechanisms.

#### Class Definition

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

#### Example Usage

```python
from astroml.models import GAT

model = GAT(
    input_dim=64,
    hidden_dims=[128, 64],
    output_dim=2,
    num_heads=8
)

x = torch.randn(100, 64)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

output = model(x, edge_index)
```

## Model Utilities

### ModelFactory

Factory class for creating models with different configurations.

#### Class Definition

```python
class ModelFactory:
    @staticmethod
    def create_gcn(config: GCNConfig) -> GCN
    
    @staticmethod
    def create_temporal_gcn(config: TemporalGCNConfig) -> TemporalGCN
    
    @staticmethod
    def create_anomaly_detector(config: AnomalyDetectorConfig) -> AnomalyDetector
```

#### Example Usage

```python
from astroml.models import ModelFactory, GCNConfig

# Create GCN with configuration
config = GCNConfig(
    input_dim=64,
    hidden_dims=[128, 64],
    output_dim=2,
    dropout=0.5
)

model = ModelFactory.create_gcn(config)
```

### ModelConfig

Configuration classes for different models.

#### GCNConfig

```python
@dataclass
class GCNConfig:
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    dropout: float = 0.5
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
```

#### TemporalGCNConfig

```python
@dataclass
class TemporalGCNConfig:
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    temporal_dim: int = 32
    dropout: float = 0.5
    learning_rate: float = 0.001
    time_encoding: str = "sinusoidal"
```

#### AnomalyDetectorConfig

```python
@dataclass
class AnomalyDetectorConfig:
    method: str = "autoencoder"
    threshold: float = 0.95
    model_type: str = "gcn"
    feature_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
```

## Training Infrastructure

### Trainer

Main training orchestrator for all models.

#### Class Definition

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

#### Methods

##### train()

Train the model on graph data.

**Parameters:**
- `graph` (TemporalGraph): Training graph
- `epochs` (int): Number of training epochs
- `learning_rate` (float): Learning rate
- `batch_size` (int): Batch size
- `validation_split` (float): Fraction of data for validation

**Returns:** TrainingMetrics - Training statistics

**Example:**
```python
from astroml.models import GCN, Trainer

model = GCN(input_dim=64, hidden_dims=[128, 64], output_dim=2)
trainer = Trainer(model)

metrics = trainer.train(
    graph=training_graph,
    epochs=100,
    learning_rate=0.001,
    batch_size=32
)

print(f"Training loss: {metrics.train_loss[-1]:.4f}")
print(f"Validation accuracy: {metrics.val_accuracy[-1]:.4f}")
```

##### evaluate()

Evaluate the model on test data.

**Parameters:**
- `graph` (TemporalGraph): Test graph

**Returns:** EvaluationMetrics - Evaluation statistics

##### predict()

Make predictions on graph data.

**Parameters:**
- `graph` (TemporalGraph): Graph for prediction

**Returns:** torch.Tensor - Model predictions

### TrainingMetrics

Container for training metrics.

#### Class Definition

```python
@dataclass
class TrainingMetrics:
    train_loss: List[float]
    val_loss: List[float]
    train_accuracy: List[float]
    val_accuracy: List[float]
    learning_rate: float
    epochs_trained: int
    training_time: float
```

### EvaluationMetrics

Container for evaluation metrics.

#### Class Definition

```python
@dataclass
class EvaluationMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: np.ndarray
```

## Experiment Management

### Experiment

Experiment management for reproducible ML research.

#### Class Definition

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

Run the complete experiment.

**Parameters:**
- `train_graph` (TemporalGraph): Training data
- `test_graph` (TemporalGraph): Test data

**Returns:** ExperimentResult - Complete experiment results

**Example:**
```python
from astroml.models import Experiment, GCN, GCNConfig

# Define experiment configuration
config = {
    "model_type": "gcn",
    "model_config": GCNConfig(
        input_dim=64,
        hidden_dims=[128, 64],
        output_dim=2
    ),
    "training": {
        "epochs": 100,
        "learning_rate": 0.001,
        "batch_size": 32
    }
}

# Run experiment
experiment = Experiment("fraud_detection_gcn", config)
results = experiment.run(train_graph, test_graph)

print(f"Experiment completed: {results}")
```

### ExperimentResult

Container for experiment results.

#### Class Definition

```python
@dataclass
class ExperimentResult:
    name: str
    config: Dict[str, Any]
    training_metrics: TrainingMetrics
    evaluation_metrics: EvaluationMetrics
    model_path: str
    timestamp: datetime
    seed: int
```

## Hyperparameter Optimization

### HyperparameterSearch

Hyperparameter optimization using grid search and random search.

#### Class Definition

```python
class HyperparameterSearch:
    def __init__(
        self,
        model_class: Type[nn.Module],
        param_grid: Dict[str, List[Any]],
        search_method: str = "grid_search"
    ) -> None
    
    def search(
        self,
        train_graph: TemporalGraph,
        val_graph: TemporalGraph,
        max_trials: int = 100
    ) -> SearchResults
    
    def optimize(
        self,
        graph: TemporalGraph,
        cv_folds: int = 5
    ) -> OptimizationResult
```

#### Methods

##### search()

Perform hyperparameter search.

**Parameters:**
- `train_graph` (TemporalGraph): Training data
- `val_graph` (TemporalGraph): Validation data
- `max_trials` (int): Maximum number of trials

**Returns:** SearchResults - Search results and best parameters

**Example:**
```python
from astroml.models import HyperparameterSearch, GCN

# Define parameter grid
param_grid = {
    "hidden_dims": [[64, 32], [128, 64], [256, 128]],
    "dropout": [0.3, 0.5, 0.7],
    "learning_rate": [0.001, 0.01, 0.1]
}

# Perform search
search = HyperparameterSearch(GCN, param_grid, search_method="random_search")
results = search.search(train_graph, val_graph, max_trials=50)

print(f"Best parameters: {results.best_params}")
print(f"Best score: {results.best_score:.4f}")
```

## Model Persistence

### ModelSaver

Save and load trained models.

#### Class Definition

```python
class ModelSaver:
    @staticmethod
    def save_model(
        model: nn.Module,
        path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> None
    
    @staticmethod
    def load_model(path: str) -> Tuple[nn.Module, Dict[str, Any]]
```

#### Methods

##### save_model()

Save a trained model to disk.

**Parameters:**
- `model` (nn.Module): Trained model
- `path` (str): Path to save the model
- `config` (Optional[Dict]): Model configuration

##### load_model()

Load a saved model from disk.

**Parameters:**
- `path` (str): Path to the saved model

**Returns:** Tuple[nn.Module, Dict] - Model and configuration

**Example:**
```python
from astroml.models import ModelSaver, GCN

# Save model
model = GCN(input_dim=64, hidden_dims=[128, 64], output_dim=2)
ModelSaver.save_model(model, "models/gcn_fraud_detection.pkl", {
    "input_dim": 64,
    "hidden_dims": [128, 64],
    "output_dim": 2
})

# Load model
loaded_model, config = ModelSaver.load_model("models/gcn_fraud_detection.pkl")
print(f"Loaded model with config: {config}")
```

## Performance Optimization

### ModelOptimizer

Optimize models for better performance.

#### Class Definition

```python
class ModelOptimizer:
    @staticmethod
    def quantize_model(model: nn.Module) -> nn.Module
    
    @staticmethod
    def prune_model(
        model: nn.Module,
        pruning_ratio: float = 0.1
    ) -> nn.Module
    
    @staticmethod
    def compile_model(model: nn.Module) -> nn.Module
```

#### Methods

##### quantize_model()

Quantize model for faster inference.

**Parameters:**
- `model` (nn.Module): Model to quantize

**Returns:** nn.Module - Quantized model

##### prune_model()

Prune model to reduce size.

**Parameters:**
- `model` (nn.Module): Model to prune
- `pruning_ratio` (float): Ratio of parameters to prune

**Returns:** nn.Module - Pruned model

##### compile_model()

Compile model for optimized execution.

**Parameters:**
- `model` (nn.Module): Model to compile

**Returns:** nn.Module - Compiled model

## Error Handling

### Custom Exceptions

#### ModelError

Base exception for model-related errors.

```python
class ModelError(Exception):
    """Base exception for model operations."""
    pass
```

#### TrainingError

Raised when training fails.

```python
class TrainingError(ModelError):
    """Raised when model training fails."""
    pass
```

#### InferenceError

Raised when inference fails.

```python
class InferenceError(ModelError):
    """Raised when model inference fails."""
    pass
```

#### ConfigurationError

Raised when model configuration is invalid.

```python
class ConfigurationError(ModelError):
    """Raised when model configuration is invalid."""
    def __init__(self, config_key: str, message: str):
        self.config_key = config_key
        super().__init__(f"Invalid configuration for {config_key}: {message}")
```

## Testing

### Unit Tests

```python
import pytest
import torch
from astroml.models import GCN, AnomalyDetector

class TestGCN:
    def test_gcn_forward(self):
        model = GCN(input_dim=64, hidden_dims=[32], output_dim=2)
        x = torch.randn(10, 64)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        output = model(x, edge_index)
        assert output.shape == (10, 2)
        assert torch.allclose(torch.exp(output).sum(dim=1), torch.ones(10))

class TestAnomalyDetector:
    def test_anomaly_detection(self):
        detector = AnomalyDetector(threshold=0.95)
        
        # Mock graph data
        graph = create_mock_graph()
        
        anomalies = detector.detect(graph)
        assert isinstance(anomalies, list)
```

### Integration Tests

```python
import pytest
from astroml.models import Trainer, GCN

class TestModelTraining:
    def test_end_to_end_training(self):
        model = GCN(input_dim=64, hidden_dims=[32], output_dim=2)
        trainer = Trainer(model)
        
        graph = create_training_graph()
        metrics = trainer.train(graph, epochs=5)
        
        assert len(metrics.train_loss) == 5
        assert metrics.epochs_trained == 5
```

## Usage Examples

### Complete ML Pipeline

```python
from astroml.models import GCN, Trainer, AnomalyDetector
from astroml.models.config import GCNConfig
from astroml.graph import GraphBuilder

# Build graph from blockchain data
builder = GraphBuilder()
graph = builder.build_snapshot(window_days=30)

# Configure and create model
config = GCNConfig(
    input_dim=graph.node_features.shape[1],
    hidden_dims=[128, 64],
    output_dim=2,  # Binary classification
    dropout=0.5
)

model = GCN(
    input_dim=config.input_dim,
    hidden_dims=config.hidden_dims,
    output_dim=config.output_dim,
    dropout=config.dropout
)

# Train model
trainer = Trainer(model, learning_rate=config.learning_rate)
training_metrics = trainer.train(
    graph=graph,
    epochs=100,
    validation_split=0.2
)

# Evaluate model
evaluation_metrics = trainer.evaluate(graph)

# Anomaly detection
anomaly_detector = AnomalyDetector(model=model, threshold=0.95)
anomalies = anomaly_detector.detect(graph)

print(f"Model trained with accuracy: {evaluation_metrics.accuracy:.4f}")
print(f"Detected {len(anomalies)} anomalies")
```

### Multi-Model Comparison

```python
from astroml.models import GCN, GraphSAGE, GAT, Trainer
from astroml.models.experiment import Experiment

# Define models to compare
models = {
    "GCN": GCN(input_dim=64, hidden_dims=[128, 64], output_dim=2),
    "GraphSAGE": GraphSAGE(input_dim=64, hidden_dims=[128, 64], output_dim=2),
    "GAT": GAT(input_dim=64, hidden_dims=[128, 64], output_dim=2)
}

results = {}

for name, model in models.items():
    trainer = Trainer(model)
    metrics = trainer.train(graph, epochs=50)
    results[name] = metrics

# Compare results
for name, metrics in results.items():
    print(f"{name}: Val Accuracy = {metrics.val_accuracy[-1]:.4f}")
```

---

This comprehensive documentation covers all aspects of the machine learning models module, from basic usage to advanced experimentation and optimization. The models are designed to work seamlessly with the AstroML graph data structures and provide a solid foundation for blockchain ML research.
