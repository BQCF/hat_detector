# tests/test_gate.py
# 人形门控：只有落在人体框内的 head/helmet 才算数。
from __future__ import annotations

from core.detector import Detection
from core.gate import GateConfig, PersonGate, cover_ratio, expand_box, keep_covered


def det(cls_name: str, box, cls_id: int | None = None):
    x1, y1, x2, y2 = box
    cid = cls_id if cls_id is not None else (1 if cls_name == "helmet" else 0)
    return Detection(x1, y1, x2, y2, cid, cls_name, 0.8)


def test_覆盖比例():
    assert cover_ratio((10, 10, 20, 20), (0, 0, 100, 100)) == 1.0
    assert cover_ratio((0, 0, 100, 100), (0, 0, 50, 50)) == 0.25
    assert cover_ratio((200, 200, 300, 300), (0, 0, 50, 50)) == 0.0
    # 一半在框内
    assert abs(cover_ratio((0, 0, 100, 100), (0, 0, 50, 100)) - 0.5) < 1e-6


def test_人体框外扩():
    x1, y1, x2, y2 = expand_box((100, 100, 200, 200), 0.1)
    assert (x1, y1, x2, y2) == (90, 90, 210, 210)


def test_保留人身上的框():
    head = det("head", (110, 100, 150, 140))
    outside = det("helmet", (500, 100, 540, 140))
    kept = keep_covered([head, outside], [(100, 90, 300, 400)], min_cover=0.5)
    assert kept == [head]


def test_边缘框按阈值取舍():
    partially = det("helmet", (250, 100, 350, 200))     # 一半在 (100,90)-(300,400) 里
    assert keep_covered([partially], [(100, 90, 300, 400)], min_cover=0.4) == [partially]
    assert keep_covered([partially], [(100, 90, 300, 400)], min_cover=0.6) == []


def test_没检出人时的两种策略():
    d = [det("head", (10, 10, 40, 40))]
    assert keep_covered(d, [], empty_policy="drop") == []
    assert keep_covered(d, [], empty_policy="keep") == d


def test_模型缺失时优雅降级(tmp_path):
    cfg = GateConfig(enabled=True, model_path=str(tmp_path / "nope.onnx"))
    gate = PersonGate(cfg, base_dir=tmp_path)
    assert gate.available is False
    assert "找不到" in gate.reason
    dets = [det("head", (0, 0, 20, 20))]
    assert gate.apply(None, dets) == dets        # 不可用就原样放行


def test_关闭时不做任何过滤(tmp_path):
    gate = PersonGate(GateConfig(enabled=False), base_dir=tmp_path)
    assert gate.available is False
    dets = [det("helmet", (0, 0, 20, 20))]
    assert gate.apply(None, dets) == dets
