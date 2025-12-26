# Test Suite

Unit tests for Tachion's core API wrappers and data collectors.

## Structure

```
test/
├── conftest.py          # Pytest configuration & shared fixtures
├── test_bapi.py         # Binance API wrapper tests (14 tests)
├── test_tdapi.py        # TwelveData API wrapper tests (14 tests)
├── test_oapi.py         # OANDA API wrapper tests (14 tests)
└── test_collectors.py   # Data collector tests (21 tests)
```

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest test/test_bapi.py

# Run specific test class
pytest test/test_bapi.py::TestBinanceAPI

# Run with coverage
pytest --cov=core --cov=data --cov-report=term-missing
```

## Test Coverage

Each function has 7 tests:
- **4 Basic Cases**: Normal use cases and expected behavior
- **3 Edge Cases**: Error handling, empty inputs, and boundary conditions

### API Wrapper Tests

| Module | Function | Basic | Edge |
|--------|----------|-------|------|
| `bapi.py` | `BinanceAPI()` | 4 | 3 |
| `bapi.py` | `call_specific_binance()` | 4 | 3 |
| `tdapi.py` | `TwelveDataAPI()` | 4 | 3 |
| `tdapi.py` | `call_specific_td()` | 4 | 3 |
| `oapi.py` | `OandaAPI()` | 4 | 3 |
| `oapi.py` | `call_specific_oanda()` | 4 | 3 |

### Collector Tests

| Module | Function | Basic | Edge |
|--------|----------|-------|------|
| `crypto/collector.py` | `write_data()` | 4 | 3 |
| `equities/collector.py` | `write_data()` | 4 | 3 |
| `forex/collector.py` | `write_data()` | 4 | 3 |

## Dependencies

- `pytest` - Test framework
- `pytest-cov` (optional) - Coverage reports

Install test dependencies:
```bash
pip install pytest pytest-cov
```
