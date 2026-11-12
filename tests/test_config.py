"""Tests for Pydantic configuration and data models."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from rigsim.config import (
    Episode,
    FaultScenarioConfig,
    QAThresholdsConfig,
    RigConfig,
    SensorKind,
    Session,
    SessionSource,
    StoragePricingConfig,
    Stream,
    Transform,
)


def test_tabletop_rig_config_loads_valid():
    rig_path = Path("configs/rigs/tabletop_6cam.yaml")
    assert rig_path.exists()
    with open(rig_path) as f:
        data = yaml.safe_load(f)
    rig = RigConfig.model_validate(data)
    assert rig.rig_name == "tabletop_6cam"
    assert len(rig.cameras) == 7  # 2 stereo + 4 corner + 1 wrist
    assert rig.imu is not None
    assert rig.imu.nominal_rate_hz == 200.0
    assert len(rig.sync_groups) == 4
    assert len(rig.clocks) == 4
    assert rig.scene_features is not None
    assert rig.scene_features.sync_strobe is not None
    assert rig.scene_features.charuco_board is not None


def test_handheld_rig_config_loads_valid():
    rig_path = Path("configs/rigs/handheld_stereo_imu.yaml")
    assert rig_path.exists()
    with open(rig_path) as f:
        data = yaml.safe_load(f)
    rig = RigConfig.model_validate(data)
    assert rig.rig_name == "handheld_stereo_imu"
    assert len(rig.cameras) == 2  # stereo pair
    assert rig.imu is not None
    assert rig.imu.nominal_rate_hz == 200.0


def test_fault_scenarios_load_valid():
    clean_path = Path("configs/faults/clean.yaml")
    with open(clean_path) as f:
        clean_data = yaml.safe_load(f)
    clean = FaultScenarioConfig.model_validate(clean_data)
    assert clean.scenario_name == "clean"
    assert len(clean.faults) == 0

    mixed_path = Path("configs/faults/realistic_mixed.yaml")
    with open(mixed_path) as f:
        mixed_data = yaml.safe_load(f)
    mixed = FaultScenarioConfig.model_validate(mixed_data)
    assert mixed.scenario_name == "realistic_mixed"
    assert len(mixed.faults) > 0


def test_qa_thresholds_loads_valid():
    qa_path = Path("configs/qa/default_thresholds.yaml")
    with open(qa_path) as f:
        qa_data = yaml.safe_load(f)
    thresholds = QAThresholdsConfig.model_validate(qa_data)
    assert thresholds.timing.max_drop_ratio == 0.01
    assert "T1_FRAME_DROPS" in thresholds.critical_checks


def test_storage_pricing_loads_valid():
    pricing_path = Path("configs/cost/storage_pricing.yaml")
    with open(pricing_path) as f:
        pricing_data = yaml.safe_load(f)
    pricing = StoragePricingConfig.model_validate(pricing_data)
    assert pricing.hot_storage_per_gb_month > 0.0
    assert pricing.archive_storage_per_gb_month < pricing.hot_storage_per_gb_month


def test_transform_math_and_validation():
    # Identity transform
    ident = Transform.identity()
    np.testing.assert_allclose(ident.as_numpy(), np.eye(4))

    # Construct from translation + 90 deg z rotation
    # qz = sin(45 deg) = 0.7071068, qw = cos(45 deg) = 0.7071068
    t = Transform.from_translation_quaternion(
        (1.0, 2.0, 3.0), (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5))
    )
    assert np.isclose(t.translation[0], 1.0)
    assert np.isclose(t.translation[1], 2.0)
    assert np.isclose(t.translation[2], 3.0)
    mat = t.as_numpy()
    # Rotation of [1, 0, 0] by 90 deg around z should be [0, 1, 0]
    p_sensor = np.array([1.0, 0.0, 0.0, 1.0])
    p_rig = mat @ p_sensor
    np.testing.assert_allclose(p_rig[:3], [1.0, 3.0, 3.0], atol=1e-6)

    # Invalid transform check: wrong bottom row
    with pytest.raises(ValueError):
        Transform(
            matrix=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 2.0],  # Invalid bottom row
            ],
            translation=(0.0, 0.0, 0.0),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )


def test_session_and_episode_schemas():
    stream = Stream(
        stream_id="cam0",
        kind=SensorKind.STEREO_LEFT,
        nominal_rate_hz=30.0,
        resolution=(1280, 720),
        clock_id="clock0",
        frame_id="cam0_optical",
        extrinsics=Transform.identity(),
    )
    session = Session(
        session_id="test_sess_001",
        source=SessionSource.SIM,
        rig_config_name="tabletop_6cam",
        created_at="2026-11-13T00:00:00Z",
        operator="lead_engineer",
        streams=[stream],
        mcap_path="data/test.mcap",
    )
    assert session.session_id == "test_sess_001"
    assert len(session.streams) == 1

    episode = Episode(
        episode_id="ep_001",
        source_session_id=session.session_id,
        t_start_ns=0,
        t_end_ns=10_000_000_000,
        duration_s=10.0,
        streams=[stream],
        frames_count={"cam0": 300},
        calibration_hash="abc1234",
        git_commit="HEAD",
        config_hash="def5678",
    )
    assert episode.episode_id == "ep_001"
    assert episode.duration_s == 10.0
