def test_list_cameras_structure():
    from core.sources import list_cameras
    cams = list_cameras(max_index=2)
    assert isinstance(cams, list)
    for index, label in cams:
        assert isinstance(index, int) and index >= 0
        assert isinstance(label, str) and label


def test_list_cameras_indices_unique():
    from core.sources import list_cameras
    cams = list_cameras(max_index=2)
    idxs = [i for i, _ in cams]
    assert len(idxs) == len(set(idxs))
