"""
Pytest configuration and shared fixtures.
"""

import pytest


@pytest.fixture
def sample_config() -> dict:
    """Provide a sample configuration for tests."""
    return {
        "model_type": "test",
        "batch_size": 32,
        "learning_rate": 0.001,
    }
