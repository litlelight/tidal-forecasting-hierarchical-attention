# Contributing to Tidal Forecasting Project

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

1. **Clear title**: Describe the problem concisely
2. **Steps to reproduce**: Detailed steps to reproduce the issue
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: Python version, OS, GPU info, etc.
6. **Code snippet**: Minimal code to reproduce (if applicable)

**Example:**
```
Title: Model training fails with CUDA out of memory

Steps to reproduce:
1. Run: python src/training/train.py --batch-size 64
2. Training starts but crashes after 10 iterations

Expected: Training completes successfully
Actual: CUDA OOM error

Environment:
- Python 3.10
- PyTorch 2.0.1
- CUDA 11.8
- GPU: RTX 3060 (12GB)
```

### Suggesting Enhancements

For feature requests, open an issue with:

1. **Use case**: Why is this feature needed?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Other approaches you've considered
4. **Additional context**: Screenshots, examples, etc.

### Contributing Code

We welcome code contributions! Areas where you can help:

- **New features**: Additional forecasting models, evaluation metrics
- **Bug fixes**: Resolve open issues
- **Documentation**: Improve README, add tutorials
- **Tests**: Increase test coverage
- **Performance**: Optimize training/inference speed

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/tidal-forecasting-hierarchical-attention.git
cd tidal-forecasting-hierarchical-attention

# Add upstream remote
git remote add upstream https://github.com/litflight/tidal-forecasting-hierarchical-attention.git
```

### 2. Create Development Environment

```bash
# Using conda
conda env create -f environment.yml
conda activate tidal-forecasting

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

This ensures code quality checks run before each commit.

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

**Branch naming conventions:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `test/` - Test additions/updates
- `refactor/` - Code refactoring

## Pull Request Process

### 1. Make Your Changes

- Write clear, commented code
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 2. Test Your Changes

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/

# Run linting
flake8 src/
black --check src/
mypy src/
```

### 3. Commit Your Changes

Follow conventional commit format:

```bash
git add .
git commit -m "feat: add new attention mechanism"
# or
git commit -m "fix: resolve CUDA memory leak in training loop"
# or
git commit -m "docs: update installation instructions"
```

**Commit types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Formatting changes
- `refactor:` - Code restructuring
- `test:` - Adding tests
- `chore:` - Maintenance tasks

### 4. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
# Provide clear description of changes
```

**PR Description Template:**
```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
How has this been tested?

## Checklist
- [ ] Code follows project style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests pass
- [ ] No new warnings
```

### 5. Code Review

- Respond to review comments
- Make requested changes
- Push updates to your branch (PR updates automatically)

### 6. Merge

Once approved, a maintainer will merge your PR.

## Coding Standards

### Python Style

Follow [PEP 8](https://pep8.org/) style guide:

```python
# Good
def calculate_tidal_residual(observed, predicted):
    """
    Calculate tidal residual (meteorological component).
    
    Args:
        observed (np.ndarray): Observed water levels
        predicted (np.ndarray): Predicted tides
        
    Returns:
        np.ndarray: Residual component
    """
    return observed - predicted


# Bad
def calc_res(o,p):
    return o-p
```

### Naming Conventions

- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Private methods**: `_leading_underscore`

```python
# Good
class HierarchicalAttentionModel:
    MAX_SEQUENCE_LENGTH = 168
    
    def __init__(self):
        self.hidden_dim = 128
        
    def _compute_attention_weights(self):
        pass
```

### Documentation

Use Google-style docstrings:

```python
def train_model(model, dataloader, epochs=100, lr=0.001):
    """
    Train the hierarchical attention model.
    
    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): Training data loader
        epochs (int, optional): Number of training epochs. Defaults to 100.
        lr (float, optional): Learning rate. Defaults to 0.001.
        
    Returns:
        dict: Training history with keys 'loss', 'val_loss'
        
    Raises:
        ValueError: If epochs < 1 or lr <= 0
        
    Example:
        >>> model = HierarchicalAttentionModel()
        >>> history = train_model(model, train_loader, epochs=50)
    """
    pass
```

### Type Hints

Use type hints for function signatures:

```python
from typing import List, Tuple, Optional
import numpy as np
import torch

def prepare_sequences(
    data: np.ndarray,
    sequence_length: int,
    forecast_horizon: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare input-output sequences."""
    pass
```

### Code Organization

```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Local imports
from src.models.hierarchical_attention import HierarchicalAttentionModel
from src.data.dataloader import TidalDataLoader
```

## Testing

### Writing Tests

Use pytest for testing:

```python
# tests/test_models.py
import pytest
import torch
from src.models.hierarchical_attention import HierarchicalAttentionModel

def test_model_initialization():
    """Test model can be initialized with default parameters."""
    model = HierarchicalAttentionModel()
    assert model is not None
    assert model.hidden_dim == 128

def test_model_forward_pass():
    """Test model forward pass with sample input."""
    model = HierarchicalAttentionModel(input_dim=8)
    input_tensor = torch.randn(32, 168, 8)  # batch_size=32, seq_len=168
    
    output = model(input_tensor)
    
    assert output.shape[0] == 32
    assert output.shape[1] == 168

@pytest.mark.parametrize("batch_size", [16, 32, 64])
def test_model_different_batch_sizes(batch_size):
    """Test model handles different batch sizes."""
    model = HierarchicalAttentionModel()
    input_tensor = torch.randn(batch_size, 168, 8)
    output = model(input_tensor)
    assert output.shape[0] == batch_size
```

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_models.py

# Specific test
pytest tests/test_models.py::test_model_initialization

# With coverage
pytest --cov=src --cov-report=html

# Verbose output
pytest -v
```

## Questions?

If you have questions, feel free to:

1. Open an issue with the `question` label
2. Email: 2627556529@qq.com
3. Check existing issues and discussions

## Recognition

Contributors will be acknowledged in:

- GitHub contributors page
- CONTRIBUTORS.md file
- Paper acknowledgments (for significant contributions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🙏
