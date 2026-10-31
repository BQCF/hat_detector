from core.alarm import AlarmConfig, AlarmManager
from core.detector import Detection


def _cfg(**kw):
    c = AlarmConfig(sound=False, snapshot=False, confirm_frames=3, cooldown_sec=1.0)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _head(track_id):
    return Detection(0, 0, 10, 10, 0, "head", 0.9, track_id=track_id)


def test_dedup_one_event_per_track():
    am = AlarmManager(_cfg(dedup_by_track=True))
    evs = []
    am.on_alarm = evs.append
    for _ in range(10):                      # 同一人连续 10 帧
        am.update({"head": 1}, None, [_head(1)])
    assert len(evs) == 1


def test_two_people_two_events():
    am = AlarmManager(_cfg(dedup_by_track=True))
    evs = []
    am.on_alarm = evs.append
    for _ in range(5):
        am.update({"head": 1}, None, [_head(1)])
    for _ in range(5):                       # 第二个人（新 track id）
        am.update({"head": 1}, None, [_head(2)])
    assert len(evs) == 2


def test_confirm_frames_gating():
    am = AlarmManager(_cfg(dedup_by_track=True, confirm_frames=3))
    evs = []
    am.on_alarm = evs.append
    am.update({"head": 1}, None, [_head(1)])
    am.update({"head": 1}, None, [_head(1)])
    assert len(evs) == 0                     # 才 2 帧，未确认
    am.update({"head": 1}, None, [_head(1)])
    assert len(evs) == 1


def test_fallback_without_track_id():
    # 没有 track_id 时回退到整帧计数 + 冷却
    am = AlarmManager(_cfg(dedup_by_track=True, cooldown_sec=0.05))
    evs = []
    am.on_alarm = evs.append
    for _ in range(6):
        am.update({"head": 1}, None, [Detection(0, 0, 10, 10, 0, "head", 0.9)])
    assert len(evs) >= 1


def test_disabled_no_alarm():
    am = AlarmManager(_cfg(enabled=False))
    assert am.update({"head": 1}, None, [_head(1)]) is False
