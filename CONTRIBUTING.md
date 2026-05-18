# Contributing to RBC-Polynomial-Regression

Thank you for your interest in contributing! This document describes how to
contribute code, tests, documentation, or new clinical features.

## Development Setup

```bash
git clone https://github.com/Aranya2801/RBC-Polynomial-Regression.git
cd RBC-Polynomial-Regression
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python data/raw/generate_dataset.py
```

## Branch Naming

| Type        | Pattern               |
|-------------|----------------------|
| Feature     | `feat/short-desc`    |
| Bug Fix     | `fix/short-desc`     |
| Docs        | `docs/short-desc`    |
| Refactor    | `refactor/desc`      |

## Code Standards

- Follow **PEP 8** (max line length 100)
- Write **docstrings** for all public classes/functions
- Add **unit tests** for every new feature in `tests/`
- Run `pytest tests/ -v` before opening a PR — all tests must pass

## Pull Request Checklist

- [ ] Tests added / updated
- [ ] All 29+ tests pass (`pytest tests/ -v`)
- [ ] Docstrings written
- [ ] `requirements.txt` updated if new packages added
- [ ] README updated if user-facing behavior changed

## Areas for Contribution

- New regularization methods (e.g., Elastic Net tuning)
- Additional clinical feature engineering
- Bayesian polynomial regression variant
- Streamlit dashboard UX improvements
- Additional datasets (MIMIC-III CBC, etc.)
