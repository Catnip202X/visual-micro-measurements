from pathlib import Path

from aoi_yolo.historical_best_deep_eval.cache import (
    ImagePrediction,
    PredictionRecord,
    TimingRecord,
    read_cache,
    write_cache,
)


def sample_prediction() -> ImagePrediction:
    return ImagePrediction(
        stem="0001",
        predictions=(
            PredictionRecord(
                prediction_id="0001:0",
                label="pinhole",
                confidence=0.75,
                area_px=200,
                polygon=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0)),
                kept=True,
            ),
        ),
        timing=TimingRecord(
            index=1,
            stem="0001",
            width=10,
            height=10,
            model_preprocess_ms=1.0,
            inference_ms=2.0,
            model_postprocess_ms=3.0,
            conversion_cascade_ms=4.0,
            total_production_ms=10.0,
            raw_predictions=1,
            kept_predictions=1,
        ),
    )


def test_prediction_cache_round_trip(tmp_path):
    path = tmp_path / "cache.jsonl.gz"
    identity = {"weights_sha256": "abc"}
    expected = sample_prediction()

    write_cache(path, [expected], identity)
    loaded_identity, loaded = read_cache(path)

    assert loaded_identity == identity
    assert loaded == [expected]

