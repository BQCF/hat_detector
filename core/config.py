# core/config.py
# 配置：dataclass + YAML。桌面版设置面板读写的就是这份配置。
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .paths import user_dir

BASE_DIR = user_dir()                                    # 可写根目录（打包后在 exe 旁边）
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "default.yaml"


@dataclass
class AlarmConfig:
    """报警相关设置。"""
    enabled: bool = True              # 是否启用报警
    confirm_frames: int = 3           # 连续 N 帧命中才确认（去抖，避免单帧误报）
    cooldown_sec: float = 1.0         # 两次报警通知之间的最小间隔（秒）
    sound: bool = True                # 报警时是否发声
    snapshot: bool = True             # 报警时是否抓拍存图
    snapshot_dir: str = "captures"    # 抓拍与事件表存放目录（相对项目根目录）
    events_csv: str = "captures/events.csv"
    alarm_class: str = "head"         # 触发报警的类别：head = 未戴安全帽
    max_snapshots: int = 2000         # 抓拍最多保留多少张，超了自动删最旧的（0=不限制）


@dataclass
class AppConfig:
    """程序总配置。"""
    model_path: str = ""              # 留空 = 自动找 runs/detect 下最新的 best.pt / best.onnx
    backend: str = "auto"             # auto / torch / onnx
    device: str = "auto"              # auto / cpu / 0 / 1 ...
    conf: float = 0.25                # 置信度阈值
    iou: float = 0.45                 # NMS IoU 阈值
    imgsz: int = 640                  # 推理分辨率
    max_det: int = 50                 # 每帧最多保留的检测框
    source_type: str = "camera"       # camera / video / image / rtsp
    source_value: str = "0"           # 摄像头序号 / 文件路径 / rtsp 地址
    camera_width: int = 1280
    camera_height: int = 720
    video_loop: bool = True           # 视频播完是否循环
    ui_interval_ms: int = 33          # 界面刷新间隔（约 30 FPS）
    alarm: AlarmConfig = field(default_factory=AlarmConfig)

    def resolve(self, p: str | Path) -> Path:
        """把配置里的相对路径按项目根目录解析成绝对路径。"""
        q = Path(p)
        return q if q.is_absolute() else (BASE_DIR / q)


def load_config(path: Path | str | None = None) -> AppConfig:
    """读取 YAML 配置；文件不存在就返回默认配置。"""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        return AppConfig()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    alarm_data = data.pop("alarm", {}) or {}
    known = {f for f in AppConfig.__dataclass_fields__ if f != "alarm"}
    cfg = AppConfig(**{k: v for k, v in data.items() if k in known})
    alarm_known = set(AlarmConfig.__dataclass_fields__)
    cfg.alarm = AlarmConfig(**{k: v for k, v in alarm_data.items() if k in alarm_known})
    return cfg


def save_config(cfg: AppConfig, path: Path | str | None = None) -> Path:
    """写回 YAML 配置，返回写入路径。"""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(asdict(cfg), allow_unicode=True, sort_keys=False),
                 encoding="utf-8")
    return p
