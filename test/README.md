# Test Suite

Unit tests for all scripts. Will expand as needed.

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

Files from the `core/` folder. We'll trust that the API is returning the correct values (no way to test that here). These tests ensure that the wrapper functions respond with the expected behaviour.

|   Module   |          Function          |
|------------|--------------------------- |
| `bapi.py`  | `BinanceAPI()`             |
| `bapi.py`  | `call_specific_binance()`  |
| `tdapi.py` | `TwelveDataAPI()`          |
| `tdapi.py` | `call_specific_td()`       |
| `oapi.py`  | `OandaAPI()`               |
| `oapi.py`  | `call_specific_oanda()`    |

### Collector Tests

Files from the `data` folder.

|         Module          |    Function     |
|-------------------------|---------------- |
| `crypto/collector.py`   | `write_data()`  |
| `equities/collector.py` | `write_data()`  |
| `forex/collector.py`    | `write_data()`  |

## Dependencies

- `pytest` - Test framework
- `pytest-cov` (optional) - FOr coverage reports

Install test dependencies:

```bash
pip install pytest pytest-cov
```
