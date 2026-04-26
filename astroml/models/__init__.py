"""AstroML Models Module"""

from .gcn import GCN
from .temporal import (
    TemporalGCN,
    TemporalGraphSAGE,
    TemporalGAT,
    TemporalGraphTransformer,
    TemporalEdgeConv,
    TemporalEncoding,
    TemporalAttention,
    TemporalModelFactory
)

__all__ = [
    'GCN',
    'TemporalGCN',
    'TemporalGraphSAGE',
    'TemporalGAT',
    'TemporalGraphTransformer',
    'TemporalEdgeConv',
    'TemporalEncoding',
    'TemporalAttention',
    'TemporalModelFactory'
]
"""Machine learning models for AstroML."""

try:
    from .deep_svdd import DeepSVDD, DeepSVDDNetwork
    from .deep_svdd_trainer import DeepSVDDTrainer, FraudDetectionDeepSVDD
except ImportError:
    pass

try:
    from .gcn import GCN
except ImportError:
    pass

from .sage_encoder import InductiveSAGEEncoder
from .deep_svdd import DeepSVDD, DeepSVDDNetwork
from .deep_svdd_trainer import DeepSVDDTrainer, FraudDetectionDeepSVDD
from .gcn import GCN
from .link_prediction import LinkPredictor, GCNEncoder

__all__ = [
    'DeepSVDD',
    'DeepSVDDNetwork',
    'DeepSVDDTrainer',
    'FraudDetectionDeepSVDD',
    'GCN',
    'InductiveSAGEEncoder',
    'GCNEncoder',
    'LinkPredictor',
]
