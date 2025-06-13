from .dual_stream_probe import DualStreamAttentionProbe
from .quantum_probe import QuantumDualStreamProbe
from .bayesian_quantum_probe import QuantumBayesianProbe

class ProbeFactory:
    """探针工厂类，用于创建不同类型的探针"""
    
    @staticmethod
    def create_probe(probe_type: str, **kwargs):
        """
        创建探针实例
        
        Args:
            probe_type: 探针类型，可选值：
                - 'dual_stream': 双流注意力探针
                - 'quantum_dual_stream': 量子双流探针
                - 'bayesian_quantum': 量子贝叶斯混合探针
            **kwargs: 探针初始化参数
        """
        if probe_type == 'dual_stream':
            return DualStreamAttentionProbe(**kwargs)
        elif probe_type == 'quantum_dual_stream':
            return QuantumDualStreamProbe(**kwargs)
        elif probe_type == 'bayesian_quantum':
            return QuantumBayesianProbe(**kwargs)
        else:
            raise ValueError(f"未知的探针类型: {probe_type}")

    @staticmethod
    def get_available_probes():
        return ['dual_stream', 'quantum_dual_stream', 'bayesian_quantum'] 