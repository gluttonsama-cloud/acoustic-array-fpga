from pathlib import Path

import pytest

from acoustic_array.core.config import ArrayConfig
from acoustic_array.io.artifacts import load_json


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def array(project_root):
    return ArrayConfig.from_mapping(load_json(project_root / "configs/array/ula16_v1.json"))
