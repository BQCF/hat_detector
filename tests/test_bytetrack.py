import itertools

import numpy as np

from core.bytetrack import ByteTracker, KalmanFilter, linear_sum_assignment
from core.detector import Detection


def _d(cx, cy, w, h, conf, cls="head"):
    return Detection(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
                     0 if cls == "head" else 1, cls, conf)


# ---------------- 匈牙利 ----------------
def test_hungarian_known():
    cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]], dtype=float)
    r, c = linear_sum_assignment(cost)
    assert sum(cost[i, j] for i, j in zip(r, c)) == 5.0


def test_hungarian_bruteforce_integer():
    for n in range(1, 6):
        for _ in range(25):
            cost = np.random.randint(0, 10, (n, n)).astype(float)
            r, c = linear_sum_assignment(cost)
            got = sum(cost[i, j] for i, j in zip(r, c))
            best = min(sum(cost[i, p[i]] for i in range(n))
                       for p in itertools.permutations(range(n)))
            assert abs(got - best) < 1e-9


def test_hungarian_rectangular():
    cost = np.array([[1, 5, 3], [2, 1, 9]], dtype=float)   # 2 行 3 列
    r, c = linear_sum_assignment(cost)
    assert len(r) == 2
    assert sum(cost[i, j] for i, j in zip(r, c)) == 2.0


# ---------------- 卡尔曼 ----------------
def test_kalman_predict_update():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([5.0, 5.0, 1.0, 10.0]))
    m2, c2 = kf.predict(mean, cov)
    assert m2[0] == 5.0 and m2[4] == 0.0      # 零速度，位置不变
    m3, _ = kf.update(m2, c2, np.array([6.0, 6.0, 1.0, 10.0]))
    assert 5.0 < m3[0] < 6.0                   # 朝量测方向移动


# ---------------- ByteTracker ----------------
def test_stable_id_across_frames():
    bt = ByteTracker()
    bt.update([_d(100, 100, 50, 50, 0.9)])
    d = [_d(101, 101, 50, 50, 0.9)]
    bt.update(d)
    assert d[0].track_id == 1


def test_new_track_new_id():
    bt = ByteTracker()
    bt.update([_d(100, 100, 50, 50, 0.9)])
    d = [_d(400, 400, 50, 50, 0.9)]
    bt.update(d)
    assert d[0].track_id == 2


def test_class_aware_no_cross_match():
    bt = ByteTracker()
    bt.update([_d(100, 100, 50, 50, 0.9, "head")])
    d = [_d(100, 100, 50, 50, 0.9, "helmet")]
    bt.update(d)
    assert d[0].track_id == 2                 # 类别不同，不续 id


def test_low_score_recovers_existing_track():
    bt = ByteTracker()
    bt.update([_d(100, 100, 50, 50, 0.9)])
    d = [_d(100, 100, 50, 50, 0.3)]           # 低置信（< track_thresh 0.5）
    bt.update(d)
    assert d[0].track_id == 1                 # 阶段2 回收


def test_low_score_alone_creates_no_track():
    bt = ByteTracker()
    d = [_d(100, 100, 50, 50, 0.3)]           # 无已有轨迹的低置信框
    bt.update(d)
    assert d[0].track_id == -1                # 不建新轨迹（阶段3只用 high）


def test_occlusion_recovers_same_id():
    bt = ByteTracker()
    bt.update([_d(100, 100, 50, 50, 0.9)])
    bt.update([])
    bt.update([])
    d = [_d(102, 102, 50, 50, 0.9)]
    bt.update(d)
    assert d[0].track_id == 1                 # 短暂遮挡后同一 id


def test_removed_after_long_loss():
    bt = ByteTracker(max_time_lost=3)
    bt.update([_d(100, 100, 50, 50, 0.9)])
    for _ in range(5):
        bt.update([])
    d = [_d(100, 100, 50, 50, 0.9)]
    bt.update(d)
    assert d[0].track_id >= 2                 # 旧轨迹已回收，给新 id


def test_stats_counts_visible():
    bt = ByteTracker()
    bt.update([_d(100, 100, 50, 50, 0.9), _d(200, 200, 50, 50, 0.9, "helmet")])
    assert bt.stats() == {"head": 1, "helmet": 1}
