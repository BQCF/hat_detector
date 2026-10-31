from core.detector import Detection
from core.roi import RoiRegion


def _det_px(cx_norm, cy_norm, w=1000, h=1000):
    """按归一化中心点构造一个像素坐标的检测框（宽高各占 10%）。"""
    cx, cy = cx_norm * w, cy_norm * h
    bw, bh = 0.1 * w, 0.1 * h
    return Detection(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, 0, "head", 0.9)


def test_inactive_region_is_noop():
    roi = RoiRegion([(0, 0), (1, 0), (1, 1), (0, 1)], enabled=False)
    assert roi.active is False
    assert roi.contains_norm(0.5, 0.5) is True


def test_requires_three_points():
    assert RoiRegion([(0, 0), (1, 1)], enabled=True).active is False


def test_point_in_polygon_left_half():
    roi = RoiRegion([(0, 0), (0.5, 0), (0.5, 1), (0, 1)], enabled=True)
    assert roi.contains_norm(0.25, 0.5) is True
    assert roi.contains_norm(0.49, 0.5) is True
    assert roi.contains_norm(0.51, 0.5) is False
    assert roi.contains_norm(0.75, 0.5) is False


def test_partition_left_half():
    roi = RoiRegion([(0, 0), (0.5, 0), (0.5, 1), (0, 1)], enabled=True)
    inside, outside = roi.partition(
        [_det_px(0.2, 0.5), _det_px(0.8, 0.5)], 1000, 1000)
    assert len(inside) == 1 and len(outside) == 1
    assert inside[0].x1 < outside[0].x1


def test_partition_resolution_independent():
    roi = RoiRegion([(0, 0), (0.5, 0), (0.5, 1), (0, 1)], enabled=True)
    for w, h in [(1000, 1000), (1920, 1080), (640, 360)]:
        inside, outside = roi.partition([_det_px(0.2, 0.5, w, h)], w, h)
        assert len(inside) == 1 and len(outside) == 0
