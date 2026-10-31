import time
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _model():
    m = ROOT / "models" / "best.onnx"
    if not m.exists():
        pytest.skip("没有 models/best.onnx，跳过集成测试")
    return str(m)


def _sample_image():
    d = ROOT / "data" / "dataset_2cls" / "images" / "val"
    imgs = sorted(d.glob("*.jpg")) if d.is_dir() else []
    return str(imgs[0]) if imgs else None


def test_onnx_detector_infers():
    img = _sample_image()
    if img is None:
        pytest.skip("没有验证集图片")
    from core.detector import Detector
    det = Detector(model_path=_model(), backend="onnx", conf=0.25)
    dets = det.infer(cv2.imread(img))
    assert det.names == {0: "head", 1: "helmet"}
    assert all(0 <= d.x1 <= d.x2 for d in dets)
    assert all(0.25 <= d.conf <= 1.0 for d in dets)


def test_pipeline_end_to_end_with_bytetrack():
    img = _sample_image()
    if img is None:
        pytest.skip("没有验证集图片")
    from core.alarm import AlarmConfig, AlarmManager
    from core.bytetrack import ByteTracker
    from core.detector import Detector
    from core.pipeline import InferPipeline
    from core.sources import create_source

    det = Detector(model_path=_model(), backend="onnx", conf=0.25)
    src = create_source("image", img)
    src.open()
    alarm = AlarmManager(AlarmConfig(sound=False, snapshot=False, dedup_by_track=True))
    pipe = InferPipeline(det, src, alarm=alarm, dedup=True, tracker=ByteTracker())
    pipe.start()
    for _ in range(500):
        if pipe.finished or pipe.error:
            break
        time.sleep(0.01)
    assert pipe.error == ""
    res = pipe.latest()
    assert res is not None and res.frame is not None
    assert res.track_stats is not None             # 有跟踪结果
    assert len(res.detections) == sum(res.counts.values())
    pipe.stop()
