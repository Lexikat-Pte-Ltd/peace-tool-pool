"""YOLOv10 adapter: detections must carry confidence.

The vendored PEACE fork's predict() returns a lossy name-keyed dict (no scores)
whenever save/save_txt/verbose/show is truthy (save_txt defaults True). Forcing
those off makes it return Results objects whose boxes carry .conf, which the
adapter then preserves.
"""

import numpy as np
import pytest

from peace_tool_pool.map_processing.detectors.peace_yolov10 import (
    COMPONENT_CLASS_NAMES,
    _YoloV10Detector,
    normalize_yolov10_result,
)


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.array(xyxy, dtype=float)
        self.conf = np.array(conf, dtype=float)
        self.cls = np.array(cls, dtype=float)


class _FakeResults:
    def __init__(self, boxes):
        self.boxes = boxes


class _RecordingModel:
    """Stand-in for the YOLOv10 model that records predict kwargs."""

    def __init__(self, results):
        self._results = results
        self.kwargs = None

    def predict(self, source, **kwargs):
        self.kwargs = kwargs
        return self._results


def _detector_with_model(model):
    det = _YoloV10Detector.__new__(_YoloV10Detector)
    det.model = model
    det.class_names = COMPONENT_CLASS_NAMES
    return det


def test_normalize_results_object_carries_confidence():
    res = _FakeResults(_FakeBoxes([[0, 0, 10, 10], [5, 5, 20, 20]], [0.91, 0.42], [1, 2]))
    out = normalize_yolov10_result(res, COMPONENT_CLASS_NAMES)
    assert out["main_map"][0].confidence == pytest.approx(0.91)
    assert out["legend"][0].confidence == pytest.approx(0.42)
    assert [round(v) for v in out["main_map"][0].bbox] == [0, 0, 10, 10]


def test_detect_disables_save_paths_so_results_carry_confidence():
    model = _RecordingModel([_FakeResults(_FakeBoxes([[0, 0, 10, 10]], [0.9], [1]))])
    det = _detector_with_model(model)

    out = det.detect("whatever.png")

    # The fix: these flags force the fork to return Results, not the lossy dict.
    assert model.kwargs.get("save_txt") is False
    assert model.kwargs.get("save") is False
    assert model.kwargs.get("verbose") is False
    # And confidence survives end-to-end.
    assert out["main_map"][0].confidence == pytest.approx(0.9)
