"""
Pytest configuration and test client fixtures.
"""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure backend app is on sys.path
backend_path = Path(__file__).resolve().parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
