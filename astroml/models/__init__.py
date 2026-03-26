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

__all__ = [
    'DeepSVDD',
    'DeepSVDDNetwork',
    'DeepSVDDTrainer',
    'FraudDetectionDeepSVDD',
    'GCN',
    'InductiveSAGEEncoder',
]
