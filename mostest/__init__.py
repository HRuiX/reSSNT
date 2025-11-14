"""
MOSTest: 多目标优化语义分割测试框架
Multi-Objective Semantic Segmentation Testing Framework
"""

__version__ = "1.0.0"

# 延迟导入 MOSTest，避免循环导入问题
# Lazy import MOSTest to avoid circular import issues
def __getattr__(name):
    if name == "MOSTest":
        from .main import MOSTest
        return MOSTest
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["MOSTest"]
