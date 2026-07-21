import hashlib
import json

from aoi_yolo.historical_best_deep_eval.run_experiment import (
    configure_ultralytics_environment,
    validate_manifest_lock,
)


def test_manifest_lock_validates_identity_and_hashes(tmp_path):
    manifest = tmp_path / "run_1.csv"
    manifest.write_text("stem\n0001\n", encoding="utf-8")
    identity = {"weights_sha256": "abc"}
    lock = {
        "experiment": identity,
        "manifests": {"run_1.csv": hashlib.sha256(manifest.read_bytes()).hexdigest()},
    }
    lock_path = tmp_path / "manifest_lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    validate_manifest_lock(lock_path, identity)
    manifest.write_text("stem\n0002\n", encoding="utf-8")
    try:
        validate_manifest_lock(lock_path, identity)
    except ValueError as exc:
        assert "hash" in str(exc)
    else:
        raise AssertionError("mutated manifest was accepted")


def test_ultralytics_config_is_redirected_to_writable_workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)
    configured = configure_ultralytics_environment(tmp_path)
    assert configured == tmp_path / "ultralytics_config"
    assert configured.exists()
