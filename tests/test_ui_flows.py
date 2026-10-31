import pytest
from PySide6.QtWidgets import QApplication, QMessageBox


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    return app or QApplication([])


def _mk_multi():
    from core.config import AppConfig
    cfg = AppConfig()
    cfg.cameras = [
        {"name": "大门", "source_type": "rtsp", "source_value": "rtsp://a"},
        {"name": "车间", "source_type": "camera", "source_value": "0"},
    ]
    return cfg


def test_exit_multi_mode(qapp, monkeypatch):
    import ui.main_window as mw
    cfg = _mk_multi()
    monkeypatch.setattr(mw, "save_config", lambda *a, **k: None)
    w = mw.MainWindow(cfg)
    assert w._multi and w.btn_exit_multi.text() == "退出多路"
    w._exit_multi_mode()
    assert w._multi is False
    assert cfg.cameras == []
    # 保留选中的第一路（大门 rtsp）作单路源
    assert cfg.source_type == "rtsp" and cfg.source_value == "rtsp://a"
    assert w.sources is not None and hasattr(w, "btn_multi")


def test_remove_stream(qapp, monkeypatch):
    import ui.main_window as mw
    cfg = _mk_multi()
    monkeypatch.setattr(mw, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(mw.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    w = mw.MainWindow(cfg)
    assert w.channels.names() == ["大门", "车间"]
    w._remove_stream("大门")
    assert [c.get("name") for c in cfg.cameras] == ["车间"]
    assert w.channels.names() == ["车间"]
    assert w._selected == "车间"       # 被删的是当前选中，切到剩下那个


def test_remove_stream_cancelled(qapp, monkeypatch):
    import ui.main_window as mw
    cfg = _mk_multi()
    monkeypatch.setattr(mw, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(mw.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    w = mw.MainWindow(cfg)
    w._remove_stream("大门")
    assert [c.get("name") for c in cfg.cameras] == ["大门", "车间"]   # 取消不删
