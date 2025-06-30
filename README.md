# amok

_Currently WIP._

Python framework to create "Agents" of certain types.

The name is a play on the image of LLMs as little minions running amok.

# Details.

There are currently the below types of agents:
1. Action Agent - An agent that can perform actions based on given commands and input.
2. Option Agent - An agent that can selects options based on given commands and input.

You can create your own agent type by sub-classing the `BaseAgent`.

## Create Agent from a Config File.
You can easily spin up agents from a config file.

The currently supported config file formats are `yaml`, `json`, and `toml`.

The fields in the config file are based on the Agent Options class for your agent type.

### Example

**config.toml**
```toml
base_url = "http://localhost:1234/v1"
model = "google/gemma-3-12b"
temperature = 0.1
max_tokens = 4000
description = "Parse messages to extract the username from them, if present."
commands = [
    "You will parse the body of the message to extract the username.",
    "ONLY reply with the username, nothing else.",
    "No other information is needed. No formatting, no explanations, nothing else!",
    "You need to be > 99% sure about your answer.",
    "Reply with 'NO_RESULT' in case no username is given or you arent sure.",
]
```

**Python Code**
```python
from amok import ActionAgent
uname_agent = ActionAgent.from_cfg("./config.toml")
```


# Development

1. Create a virtual environment (recommended):
   ```sh
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```sh
   pip install -e .
   ```

## Requirements

- Python 3.12 or newer is required.

## Running Tests

To Run tests, install `pytest` and `pytest-cov` first.
```shell
# With Pytest configured in the pyproject.toml, you can run:
pytest .

# Optional
pytest --cov=amok --cov-report=term-missing tests
```

## Running Linters
```shell
black .

ruff check
```
