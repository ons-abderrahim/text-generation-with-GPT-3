# Contributing to gpt-finetune

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/yourusername/gpt-finetune.git
cd gpt-finetune
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Code Style

- Follow PEP 8
- Use type hints for all public functions
- Write docstrings for all public functions and classes (Google style)
- Maximum line length: 100 characters

## Pull Request Guidelines

1. Fork the repo and create your branch from `main`
2. Add tests for any new functionality
3. Ensure all tests pass
4. Update documentation if needed
5. Open a PR with a clear description of changes

## Reporting Issues

Please include:
- Python version
- GPU type and CUDA version (if applicable)
- Exact error message and stack trace
- Minimal reproducible example
