import sys
import pathlib
import types

if "pyarrow" not in sys.modules:
    _pa = types.ModuleType("pyarrow")
    _pa.lib = types.ModuleType("pyarrow.lib")
    _pa.__version__ = "0.0.0"
    sys.modules["pyarrow"] = _pa
    sys.modules["pyarrow.lib"] = _pa.lib

ROOT = pathlib.Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient
import torch
from service import api as service_api


@pytest.fixture
def client():
    service_api.state = service_api.ModelState()
    return TestClient(service_api.app)


@pytest.fixture
def cpu_device():
    return torch.device("cpu")

