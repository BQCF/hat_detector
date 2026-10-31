import tempfile

from core.config import AppConfig, load_config, save_config


def test_defaults():
    cfg = AppConfig()
    assert cfg.alarm.dedup_by_track is True
    assert cfg.track.enabled is True
    assert cfg.roi.only_inside is True
    assert cfg.cameras == []


def test_roundtrip_nested():
    cfg = AppConfig()
    cfg.conf = 0.4
    cfg.alarm.confirm_frames = 5
    cfg.roi.enabled = True
    cfg.roi.points = [[0.1, 0.2], [0.9, 0.2], [0.5, 0.9]]
    cfg.cameras = [{"name": "大门", "source_type": "rtsp", "source_value": "rtsp://x"}]
    with tempfile.TemporaryDirectory() as d:
        p = save_config(cfg, f"{d}/c.yaml")
        c2 = load_config(p)
    assert c2.conf == 0.4
    assert c2.alarm.confirm_frames == 5
    assert c2.roi.enabled and c2.roi.points[0] == [0.1, 0.2]
    assert c2.cameras[0]["name"] == "大门"


def test_load_missing_returns_default():
    cfg = load_config("D:/projects/hat_detector/_does_not_exist.yaml")
    assert cfg.conf == 0.25
