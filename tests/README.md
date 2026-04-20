# Tests Directory

Store unit tests and integration tests here using pytest.

## Naming Convention

Use the format: `test_<module>.py`

Example: `test_data_loader.py`, `test_model_utils.py`

## Running Tests

```bash
# Run all tests
make test

# Or use pytest directly
pytest

# With verbose output
pytest -v
```

## Guidelines

- Write tests for data processing, models, and utilities
- Follow pytest conventions and use fixtures for setup
- Aim for >80% code coverage in the main package
