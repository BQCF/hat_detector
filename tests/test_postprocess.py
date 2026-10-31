from core.detector import Detection
from core.postprocess import _iou, count_classes, cross_class_dedup


def test_count_classes():
    dets = [
        Detection(0, 0, 10, 10, 0, "head", 0.9),
        Detection(0, 0, 10, 10, 0, "head", 0.8),
        Detection(0, 0, 10, 10, 1, "helmet", 0.9),
    ]
    assert count_classes(dets) == {"head": 2, "helmet": 1}


def test_iou_quarter():
    a = Detection(0, 0, 10, 10, 0, "head", 0.9)
    b = Detection(5, 5, 15, 15, 0, "head", 0.9)
    # 交 25，并 100+100-25=175
    assert abs(_iou(a, b) - 25 / 175) < 1e-6


def test_iou_disjoint_zero():
    a = Detection(0, 0, 10, 10, 0, "head", 0.9)
    b = Detection(50, 50, 60, 60, 0, "head", 0.9)
    assert _iou(a, b) == 0.0


def test_cross_class_dedup_keeps_higher_conf():
    head = Detection(0, 0, 100, 100, 0, "head", 0.95)
    helmet = Detection(0, 0, 100, 100, 1, "helmet", 0.90)
    kept = cross_class_dedup([head, helmet])
    assert len(kept) == 1 and kept[0].cls_name == "head"


def test_cross_class_dedup_keeps_distinct():
    a = Detection(0, 0, 10, 10, 0, "head", 0.9)
    b = Detection(50, 50, 60, 60, 1, "helmet", 0.9)
    assert len(cross_class_dedup([a, b])) == 2
