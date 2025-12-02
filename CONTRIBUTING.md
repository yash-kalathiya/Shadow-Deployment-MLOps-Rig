# Contributing to Shadow Deployment & Drift Detection Platform

First off, thank you for considering contributing to this project! 🎉

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Release Process](#release-process)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to fostering an open and welcoming environment. By participating, you are expected to uphold this by treating all contributors with respect.

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git

### Setting Up Development Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Shadow-Deployment-MLOps-Rig.git
   cd Shadow-Deployment-MLOps-Rig
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   make install-dev
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Verify the setup**
   ```bash
   make test
   make lint
   ```

## Development Workflow

### Branching Strategy

We use a simplified Git Flow:

```
main (production-ready)
  └── develop (integration branch)
       ├── feature/your-feature-name
       ├── bugfix/issue-description
       └── hotfix/critical-fix
```

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Making Changes

1. **Write your code** following our [coding standards](#coding-standards)
2. **Add tests** for new functionality
3. **Update documentation** if needed
4. **Run quality checks**:
   ```bash
   make quality  # Runs lint, format-check, typecheck
   make test     # Runs all tests
   ```

### Committing Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(api): add batch prediction endpoint"
git commit -m "fix(drift): handle empty feature arrays in PSI calculation"
git commit -m "docs: update API documentation with examples"
```

## Pull Request Process

### Before Submitting

1. ✅ All tests pass (`make test`)
2. ✅ Code is formatted (`make format`)
3. ✅ No linting errors (`make lint`)
4. ✅ Type hints are correct (`make typecheck`)
5. ✅ Documentation is updated
6. ✅ CHANGELOG.md is updated (if applicable)

### PR Template

When creating a PR, please include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How has this been tested?

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review
- [ ] I have added tests for my changes
- [ ] All new and existing tests pass
- [ ] I have updated the documentation
```

### Review Process

1. At least one maintainer must approve the PR
2. All CI checks must pass
3. No unresolved comments
4. Branch must be up-to-date with develop

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with these additions:

```python
# ✅ Good: Descriptive names, type hints, docstrings
def calculate_drift_score(
    reference: np.ndarray,
    current: np.ndarray,
    method: DriftMethod = DriftMethod.PSI,
) -> float:
    """
    Calculate drift score between two distributions.
    
    Args:
        reference: Reference distribution array
        current: Current distribution array
        method: Statistical method to use
        
    Returns:
        Drift score between 0 and 1
        
    Raises:
        ValueError: If arrays are empty
    """
    ...

# ❌ Bad: Unclear names, no types, no docstring
def calc(a, b, m=1):
    ...
```

### Import Order

```python
# Standard library
import json
import logging
from pathlib import Path

# Third-party
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Local
from src.config import settings
from src.models import ChampionModel
```

### Class Structure

```python
class MyClass:
    """Class docstring."""
    
    # Class attributes (with type hints)
    CONSTANT: ClassVar[str] = "value"
    
    def __init__(self, param: str) -> None:
        """Initialize with description."""
        self._private = param
        
    @property
    def public_property(self) -> str:
        """Property docstring."""
        return self._private
        
    def public_method(self) -> None:
        """Method docstring."""
        pass
        
    def _private_method(self) -> None:
        """Private method docstring."""
        pass
```

### Error Handling

```python
# ✅ Good: Specific exceptions with context
try:
    result = model.predict(features)
except ModelNotLoadedError as e:
    logger.error(f"Model not loaded: {e}", extra={"model": model.name})
    raise
except PredictionError as e:
    logger.warning(f"Prediction failed: {e}")
    return fallback_prediction()

# ❌ Bad: Bare except, swallowing errors
try:
    result = model.predict(features)
except:
    pass
```

## Testing Guidelines

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── test_api.py           # API endpoint tests
├── test_models.py        # Model unit tests
├── test_drift.py         # Drift detection tests
├── test_integration.py   # Integration tests
└── fixtures/
    └── sample_data.json  # Test data
```

### Writing Tests

```python
import pytest
from src.models import ChampionModel

class TestChampionModel:
    """Tests for ChampionModel."""
    
    @pytest.fixture
    def model(self) -> ChampionModel:
        """Create a loaded model instance."""
        model = ChampionModel()
        model.load()
        return model
    
    def test_predict_returns_valid_probability(self, model: ChampionModel) -> None:
        """Prediction probability should be between 0 and 1."""
        features = {"tenure": 24, "monthly_charges": 75}
        result = model.predict(features)
        
        assert 0 <= result.probability <= 1
        
    def test_predict_raises_on_invalid_features(self, model: ChampionModel) -> None:
        """Should raise FeatureValidationError for invalid input."""
        with pytest.raises(FeatureValidationError, match="Invalid value"):
            model.predict({"tenure": "invalid"})
            
    @pytest.mark.parametrize("tenure,expected_risk", [
        (60, "LOW"),
        (12, "MEDIUM"),
        (1, "HIGH"),
    ])
    def test_risk_tiers(
        self, 
        model: ChampionModel, 
        tenure: int, 
        expected_risk: str
    ) -> None:
        """Risk tier should match tenure patterns."""
        result = model.predict({"tenure": tenure})
        assert result.risk_tier == expected_risk
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_models.py -v

# Run tests matching pattern
pytest -k "test_predict" -v
```

## Documentation

### Docstring Format

We use Google-style docstrings:

```python
def function_name(param1: str, param2: int = 0) -> dict[str, Any]:
    """
    Short description of function.
    
    Longer description if needed. Can span multiple lines
    and include details about the algorithm or behavior.
    
    Args:
        param1: Description of param1
        param2: Description of param2. Defaults to 0.
        
    Returns:
        Dictionary containing:
            - key1: Description
            - key2: Description
            
    Raises:
        ValueError: If param1 is empty
        TypeError: If param2 is not an integer
        
    Example:
        >>> result = function_name("test", 42)
        >>> print(result["key1"])
        "expected output"
        
    Note:
        Any additional notes or warnings.
    """
```

### Updating Documentation

1. Update docstrings in code
2. Update README.md for user-facing changes
3. Update API documentation if endpoints change
4. Add to CHANGELOG.md

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):

- MAJOR: Breaking API changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Creating a Release

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release commit:
   ```bash
   git commit -m "chore: release v1.2.0"
   ```
4. Create and push tag:
   ```bash
   make release VERSION=1.2.0
   ```
5. GitHub Actions will automatically:
   - Run tests
   - Build Docker image
   - Create GitHub release

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Reach out to maintainers for urgent matters

Thank you for contributing! 🚀
