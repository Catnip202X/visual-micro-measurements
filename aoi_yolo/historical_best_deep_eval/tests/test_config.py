from pathlib import Path

from aoi_yolo.historical_best_deep_eval.config import (
    ThresholdRule,
    experiment_identity,
    load_config,
)


def test_frozen_operating_point():
    config_path = Path(__file__).parents[1] / "config.yaml"
    config = load_config(config_path, data_root=Path("H:/visual micro measurements"))

    assert config.imgsz == 1280
    assert config.raw_confidence == 0.05
    assert config.hit_iou == 0.05
    assert config.rules["pinhole"] == ThresholdRule(0.30, 150)
    assert config.rules["scratch"] == ThresholdRule(0.20, 300)
    assert config.run_seeds == (2026072101, 2026072102)
    assert config.images_per_run == 500
    assert config.timing_brackets_ms == (0.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, float("inf"))


def test_identity_changes_if_weights_change(tmp_path):
    config_path = Path(__file__).parents[1] / "config.yaml"
    config = load_config(config_path, data_root=tmp_path)
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    first_identity = experiment_identity(config, first)
    second_identity = experiment_identity(config, second)

    assert first_identity["weights_sha256"] != second_identity["weights_sha256"]
    assert first_identity["run_seeds"] == [2026072101, 2026072102]

