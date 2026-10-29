# core 包：检测、采集、报警的纯逻辑层，**不依赖 Qt**。
# 这样网页版、桌面版、无界面 CLI 三种入口共用同一套实现，不会出现三份逻辑。
from .config import AlarmConfig, AppConfig, load_config, save_config
from .detector import Detection, Detector
from .alarm import AlarmManager
from .pipeline import InferPipeline, PipelineResult
from .postprocess import count_classes, cross_class_dedup
from .sources import create_source, FrameSource

__all__ = [
    "AlarmConfig", "AppConfig", "load_config", "save_config",
    "Detection", "Detector",
    "AlarmManager",
    "InferPipeline", "PipelineResult",
    "count_classes", "cross_class_dedup",
    "create_source", "FrameSource",
]
