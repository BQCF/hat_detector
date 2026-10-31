# core/stabilizer.py
# 时序类别稳定器：同一 track 跨帧投票 + 迟滞，消除"一会儿带头盔一会儿不带"的闪烁。
#
# 为什么需要：单帧独立判决时，只要置信度在阈值边缘，同一颗头就会在 head/helmet
# 之间反复跳。有了 track id，就可以把"最近若干帧的判断"累积起来投票；
# 再用进入/退出两套阈值（迟滞）+ 连续 N 帧才允许改判，避免来回抖。
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

from .detector import Detection


@dataclass
class StabConfig:
    """时序投票参数。"""
    enabled: bool = True
    window: int = 15              # 参与投票的帧数（上限，实际按衰减）
    decay: float = 0.85           # 每帧旧证据的衰减系数（越小越"健忘"）
    adopt_share: float = 0.70     # 第一次定类需要的得票率
    switch_share: float = 0.80    # 已经定了类之后，改判需要的得票率（迟滞：比首次更严）
    switch_frames: int = 4        # 连续多少帧达标才真的改判
    min_obs: int = 3              # 至少观察几帧才敢下结论
    adopt_after: int = 8          # 迟迟分不出高下时，观察够这么多帧就先按多数定下来
    forget_sec: float = 3.0       # 多久没再出现就忘掉这个 track


class _TrackVote:
    __slots__ = ("scores", "seen", "stable", "streak", "last_t", "last_share")

    def __init__(self):
        self.scores: dict[int, float] = {}
        self.seen = 0
        self.stable: int | None = None
        self.streak = 0
        self.last_t = 0.0
        self.last_share = 0.0


class ClassStabilizer:
    """对带 track_id 的检测做跨帧投票，输出类别稳定的结果。"""

    def __init__(self, cfg: StabConfig | None = None, names: dict[int, str] | None = None):
        self.cfg = cfg or StabConfig()
        self.names = dict(names or {})
        self._tracks: dict[int, _TrackVote] = {}
        self._voted = 0          # 本帧被投票改写的检测数
        self._switches = 0       # 累计改判次数

    # ---------------- 对外 ----------------
    def update(self, dets: list[Detection], now: float | None = None) -> list[Detection]:
        """吃一帧检测，返回类别已被稳定化的新列表（不修改入参对象）。"""
        self._voted = 0
        if not self.cfg.enabled:
            return dets

        out: list[Detection] = []
        for det in dets:
            tid = getattr(det, "track_id", -1)
            if tid is None or tid < 0:
                out.append(det)
                continue
            vote = self._tracks.get(tid)
            if vote is None:
                vote = self._tracks[tid] = _TrackVote()
            cls = int(det.cls_id)
            for k in vote.scores:
                vote.scores[k] *= self.cfg.decay
            vote.scores[cls] = vote.scores.get(cls, 0.0) + max(float(det.conf), 1e-3)
            vote.seen += 1
            if now is not None:
                vote.last_t = now

            total = sum(vote.scores.values())
            top_cls, top_score = max(vote.scores.items(), key=lambda kv: kv[1])
            share = top_score / total if total > 0 else 0.0
            vote.last_share = share

            if vote.stable is None:
                if vote.seen >= self.cfg.min_obs and (share >= self.cfg.adopt_share
                                                      or vote.seen >= self.cfg.adopt_after):
                    vote.stable = top_cls
                    vote.streak = 0
            else:
                if top_cls != vote.stable and share >= self.cfg.switch_share:
                    vote.streak += 1
                else:
                    vote.streak = 0
                if vote.streak >= self.cfg.switch_frames:
                    vote.stable = top_cls
                    vote.streak = 0
                    self._switches += 1

            if vote.stable is not None and vote.stable != cls:
                self._voted += 1
                det = replace(det, cls_id=vote.stable,
                              cls_name=self.names.get(vote.stable, str(vote.stable)))
            out.append(det)

        self._forget(now)
        return out

    def label_of(self, track_id: int) -> int | None:
        """某个 track 当前的稳定类别（没定下来返回 None）。"""
        v = self._tracks.get(track_id)
        return None if v is None else v.stable

    def stats(self) -> dict:
        stable = Counter(v.stable for v in self._tracks.values() if v.stable is not None)
        return {
            "tracks": len(self._tracks),
            "voted_this_frame": self._voted,
            "switches": self._switches,
            "stable": {self.names.get(k, str(k)): n for k, n in stable.items()},
        }

    def reset(self) -> None:
        self._tracks.clear()
        self._voted = 0
        self._switches = 0

    # ---------------- 内部 ----------------
    def _forget(self, now: float | None) -> None:
        """忘掉久未出现的轨迹，避免长时间运行后字典无限增长。"""
        if now is None or not self._tracks:
            return
        dead = [k for k, v in self._tracks.items() if now - v.last_t > self.cfg.forget_sec]
        for k in dead:
            self._tracks.pop(k, None)
