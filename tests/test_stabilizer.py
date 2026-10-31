# tests/test_stabilizer.py
# 时序类别投票：同一 track 跨帧定类别，消除 head/helmet 来回闪。
from __future__ import annotations

from core.detector import Detection
from core.stabilizer import ClassStabilizer, StabConfig

NAMES = {0: "head", 1: "helmet"}


def det(cls_id: int, conf: float = 0.55, tid: int = 1, box=(10, 10, 60, 60)):
    x1, y1, x2, y2 = box
    return Detection(x1, y1, x2, y2, cls_id, NAMES[cls_id], conf, track_id=tid)


def run(seq, cfg=None, names=NAMES):
    st = ClassStabilizer(cfg or StabConfig(), names=names)
    out = []
    for i, c in enumerate(seq):
        out.append(st.update([det(c)], now=i * 0.1)[0].cls_name)
    return out, st


def test_稳定后不再抖动():
    """八成帧是 head：定下来以后，中间零星几帧 helmet 不应改变输出。"""
    seq = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    out, st = run(seq)
    assert out[3:] == ["head"] * (len(seq) - 3)
    assert st.stats()["switches"] == 0


def test_真实变化会被跟上():
    """前 15 帧没戴、后 15 帧戴上：应当改判一次，且不早于模型稳定之后。"""
    out, st = run([0] * 15 + [1] * 15)
    assert out[0] == "head"
    assert out[-1] == "helmet"
    assert st.stats()["switches"] == 1
    # 改判发生在切换之后的若干帧，不是立刻（迟滞）
    assert out[15:19] == ["head", "head", "head", "head"]


def test_单帧误判不改输出():
    out, _ = run([0] * 10 + [1] + [0] * 5)
    assert out[10] == "head"


def test_没有track_id的检测原样通过():
    st = ClassStabilizer(StabConfig(), names=NAMES)
    d = Detection(0, 0, 10, 10, 1, "helmet", 0.9, track_id=-1)
    out = st.update([d])[0]
    assert out is d or out.cls_name == "helmet"


def test_关闭后什么都不做():
    st = ClassStabilizer(StabConfig(enabled=False), names=NAMES)
    out = st.update([det(1)])[0]
    assert out.cls_name == "helmet"
    assert st.stats()["voted_this_frame"] == 0


def test_两个track互不干扰():
    st = ClassStabilizer(StabConfig(), names=NAMES)
    for i in range(12):
        st.update([det(0, 0.6, tid=1), det(1, 0.6, tid=2)], now=i * 0.1)
    assert st.label_of(1) == 0
    assert st.label_of(2) == 1


def test_久未出现会被遗忘():
    st = ClassStabilizer(StabConfig(forget_sec=1.0), names=NAMES)
    for i in range(10):
        st.update([det(0)], now=i * 0.1)
    assert st.stats()["tracks"] == 1
    st.update([det(1, tid=99)], now=10.0)      # 时间跳过去，旧轨迹该被清掉
    assert st.label_of(1) is None
