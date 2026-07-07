# Contributing to Wencis

First of all, thank you for taking the time to contribute! We want to make contributing to Wencis as easy and transparent as possible.

## Development Setup

To set up a local development environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/varsenai/wencis.git
   cd wencis
   ```


2. Create a virtual environment and install development dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
   pip install -e .[dev]
   ```

## Running Tests

We use `pytest` for all unit and concurrency testing. Make sure all tests pass before submitting a pull request:

```bash
python -m pytest
```

## Coding Standards

- **Formatting**: We follow standard Python PEP 8 formatting conventions.
- **Type Annotations**: All public modules and functions must be fully type-annotated. Ensure there are no type checker errors before submitting changes.
- **Commit Messages**: Write clear, descriptive commit messages.

## Submitting Pull Requests

1. Create a feature branch off `main`.
2. Implement your changes and add corresponding unit tests under `tests/`.
3. Verify all tests pass.
4. Open a pull request with a detailed description of the modifications.
