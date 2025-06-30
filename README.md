# amok

Python framework to put LLMs to work.

The name is a play on the image of LLMs as little minions running amok.

## Requirements

- Python 3.12 or newer is required.

## Setup

1. Create a virtual environment (recommended):
   ```sh
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```sh
   pip install -e .
   ```


# Running Tests

To Run tests, install `pytest` and `pytest-cov` first.
```shell
# With Pytest configured in the pyproject.toml, you can run:
pytest .

# Optional
pytest --cov=amok --cov-report=term-missing tests
```
