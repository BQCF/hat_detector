# core/bytetrack.py
# 真正的 ByteTrack 多目标跟踪器（自实现，只依赖 numpy，onnx 打包版可用）。
# 相对 core/tracker.py 的 IouTracker 升级点：
#   1. 卡尔曼滤波（8 维状态 [cx,cy,a,h,vx,vy,va,vh]）预测 + 更新，抗短暂遮挡/丢帧；
#   2. 两阶段关联：高置信检测先匹配，低置信检测再回收给剩余轨迹（ByteTrack 的核心）；
#   3. 匈牙利算法做最优一对一分配（而非贪心）。
from __future__ import annotations

import numpy as np

from .detector import Detection


def _det_tlwh(d: Detection) -> np.ndarray:
    return np.array([d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1], dtype=float)


class KalmanFilter:
    """标准 ByteTrack 卡尔曼滤波：状态 [cx, cy, a, h, vcx, vcy, va, vh]。"""

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._wp = 1.0 / 20     # 位置噪声权重
        self._wv = 1.0 / 160    # 速度噪声权重

    def initiate(self, measurement) -> tuple[np.ndarray, np.ndarray]:
        mean_pos = np.asarray(measurement, dtype=float)
        mean = np.r_[mean_pos, np.zeros(4)]
        h = mean_pos[3]
        std = [2 * self._wp * h, 2 * self._wp * h, 1e-2, 2 * self._wp * h,
               10 * self._wv * h, 10 * self._wv * h, 1e-5, 10 * self._wv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, covariance) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std_pos = [self._wp * h, self._wp * h, 1e-2, self._wp * h]
        std_vel = [self._wv * h, self._wv * h, 1e-5, self._wv * h]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean, covariance) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self._wp * h, self._wp * h, 1e-1, self._wp * h]
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return mean, covariance

    def update(self, mean, covariance, measurement) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        H = self._update_mat
        K = np.linalg.solve(projected_cov, (covariance @ H.T).T).T   # K = P H^T S^-1
        innovation = np.asarray(measurement, dtype=float) - projected_mean
        new_mean = mean + K @ innovation
        new_cov = covariance - K @ projected_cov @ K.T
        return new_mean, new_cov

    @staticmethod
    def xyah_to_tlwh(xyah) -> np.ndarray:
        ret = np.asarray(xyah, dtype=float).copy()
        ret[2] *= ret[3]           # w = a * h
        ret[:2] -= ret[2:] / 2     # 回到左上角
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh) -> np.ndarray:
        ret = np.asarray(tlwh, dtype=float).copy()
        ret[:2] += ret[2:] / 2     # 中心
        ret[2] /= ret[3]           # 宽高比 a = w / h
        return ret


class TrackState:
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """单条轨迹：卡尔曼状态 + 生命周期状态机。"""

    def __init__(self, tlwh, score, cls_name, track_id):
        self.tlwh = np.asarray(tlwh, dtype=float)
        self.score = float(score)
        self.cls_name = cls_name
        self.track_id = track_id
        self.is_activated = False
        self.state = TrackState.Tracked
        self.time_since_update = 0
        self.frame_id = 0
        self.end_frame = 0
        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(KalmanFilter.tlwh_to_xyah(self.tlwh))

    @property
    def pred_tlwh(self) -> np.ndarray:
        """卡尔曼预测的当前框（用于 IoU 距离）。"""
        return KalmanFilter.xyah_to_tlwh(self.mean[:4])

    def predict(self) -> None:
        if self.state != TrackState.Tracked:
            self.mean[7] = 0         # 未在跟踪的轨迹，速度归零
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.time_since_update += 1

    def activate(self, frame_id) -> None:
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.end_frame = frame_id

    def _absorb(self, det: Detection, frame_id) -> None:
        det.track_id = self.track_id
        self.frame_id = frame_id
        self.end_frame = frame_id
        measurement = KalmanFilter.tlwh_to_xyah(_det_tlwh(det))
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, measurement)
        self.tlwh = _det_tlwh(det)
        self.score = det.conf
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0

    def update(self, det: Detection, frame_id) -> None:
        self._absorb(det, frame_id)

    def re_activate(self, det: Detection, frame_id) -> None:
        self._absorb(det, frame_id)

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed


def _iou_tlwh(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def linear_sum_assignment(cost):
    """匈牙利算法求最小成本分配，返回 (row_ind, col_ind)。cost 可非方阵。"""
    cost = np.asarray(cost, dtype=float)
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape

    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)      # p[j] = 分配给列 j 的行（0 表示未分配）
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j] != 0:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    return (cols, rows) if transposed else (rows, cols)


class ByteTracker:
    """ByteTrack：卡尔曼 + 两阶段关联 + 匈牙利分配。

    update(dets) 给每个检测框写回 track_id，返回同一列表。
    """

    def __init__(self, track_thresh: float = 0.5, det_thresh: float = 0.2,
                 match_thresh: float = 0.3, max_time_lost: int = 30):
        self.track_thresh = float(track_thresh)
        self.det_thresh = float(det_thresh)
        self.match_thresh = float(match_thresh)
        self.max_time_lost = int(max_time_lost)
        self.frame_id = 0
        self.tracked: list[STrack] = []
        self.lost: list[STrack] = []
        self.removed: list[STrack] = []
        self._next_id = 1

    def reset(self) -> None:
        self.frame_id = 0
        self.tracked.clear()
        self.lost.clear()
        self.removed.clear()
        self._next_id = 1

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    # ---------------- 关联 ----------------
    def _iou_matrix(self, tracks, dets) -> np.ndarray:
        m = np.zeros((len(tracks), len(dets)))
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                if d.cls_name != t.cls_name:
                    m[i, j] = 0.0           # 类别不同直接不匹配
                else:
                    m[i, j] = _iou_tlwh(t.pred_tlwh, _det_tlwh(d))
        return m

    def _associate(self, tracks, dets, iou_thr):
        """返回 (matches, 未匹配轨迹下标, 未匹配检测下标)。"""
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        iou = self._iou_matrix(tracks, dets)
        cost = 1.0 - iou
        rows, cols = linear_sum_assignment(cost)
        matches = []
        u_tracks = set(range(len(tracks)))
        u_dets = set(range(len(dets)))
        for r, c in zip(rows, cols):
            if iou[r, c] >= iou_thr:
                matches.append((r, c))
                u_tracks.discard(r)
                u_dets.discard(c)
        return matches, sorted(u_tracks), sorted(u_dets)

    # ---------------- 主流程 ----------------
    def update(self, dets: list[Detection]) -> list[Detection]:
        self.frame_id += 1
        dets = [d for d in dets if d.conf >= self.det_thresh]
        high = [d for d in dets if d.conf >= self.track_thresh]
        low = [d for d in dets if d.conf < self.track_thresh]

        pool = self.tracked + self.lost
        for t in pool:
            t.predict()

        # 阶段 1：高置信检测 ↔ 所有轨迹
        matches, u_pool, u_high = self._associate(pool, high, self.match_thresh)
        refind = []
        for it, idet in matches:
            track, d = pool[it], high[idet]
            if track.state == TrackState.Tracked:
                track.update(d, self.frame_id)
            else:
                track.re_activate(d, self.frame_id)
                refind.append(track)

        # 阶段 2：低置信检测 ↔ 剩余还在跟踪的轨迹（回收被漏检/遮挡的目标）
        remain = [pool[i] for i in u_pool if pool[i].state == TrackState.Tracked]
        matches2, u_remain, _ = self._associate(remain, low, 0.5)
        for it, idet in matches2:
            remain[it].update(low[idet], self.frame_id)
        for it in u_remain:
            remain[it].mark_lost()

        # 阶段 3：未匹配的高置信检测 → 新轨迹
        for i in u_high:
            d = high[i]
            t = STrack(_det_tlwh(d), d.conf, d.cls_name, self._new_id())
            t.activate(self.frame_id)
            d.track_id = t.track_id
            self.tracked.append(t)

        # 阶段 4：回收长时间丢失的轨迹
        for t in self.lost:
            if t.state == TrackState.Lost and self.frame_id - t.end_frame > self.max_time_lost:
                t.mark_removed()
                self.removed.append(t)

        # 整理 tracked / lost / removed
        lost_this_frame = [t for t in self.tracked if t.state != TrackState.Tracked]
        self.tracked = [t for t in self.tracked if t.state == TrackState.Tracked]
        self.lost = [t for t in self.lost if t.state == TrackState.Lost]
        self.lost.extend(lost_this_frame)
        self.tracked.extend(refind)

        return dets

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.tracked:
            if t.state == TrackState.Tracked and t.time_since_update == 0:
                out[t.cls_name] = out.get(t.cls_name, 0) + 1
        return out
