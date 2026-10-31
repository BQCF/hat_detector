from core.detector import Detection
from core.tracker import IouTracker


def test_stable_id_across_frames():
    tr = IouTracker()
    d1 = [Detection(0, 0, 10, 10, 0, "head", 0.9)]
    tr.update(d1)
    assert d1[0].track_id == 1
    d2 = [Detection(0, 0, 10, 10, 0, "head", 0.9)]
    tr.update(d2)
    assert d2[0].track_id == 1                     # 同一目标 id 稳定


def test_new_track_new_id():
    tr = IouTracker()
    a = [Detection(0, 0, 10, 10, 0, "head", 0.9)]
    tr.update(a)
    b = [Detection(100, 100, 110, 110, 0, "head", 0.9)]
    tr.update(b)
    assert b[0].track_id == 2


def test_class_aware_no_cross_match():
    tr = IouTracker()
    a = [Detection(0, 0, 10, 10, 0, "head", 0.9)]
    tr.update(a)
    b = [Detection(0, 0, 10, 10, 1, "helmet", 0.9)]   # 同位置但类别不同
    tr.update(b)
    assert b[0].track_id == 2


def test_recycle_after_missing():
    tr = IouTracker(max_age=2)
    tr.update([Detection(0, 0, 10, 10, 0, "head", 0.9)])
    for _ in range(5):
        tr.update([])
    c = [Detection(0, 0, 10, 10, 0, "head", 0.9)]
    tr.update(c)
    assert c[0].track_id >= 2                       # 旧轨迹已回收，给了新 id


def test_reset():
    tr = IouTracker()
    tr.update([Detection(0, 0, 10, 10, 0, "head", 0.9)])
    tr.reset()
    d = [Detection(0, 0, 10, 10, 0, "head", 0.9)]
    tr.update(d)
    assert d[0].track_id == 1
