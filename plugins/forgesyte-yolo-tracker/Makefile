.PHONY: test-fast test-all test-heavy lint format type-check

# Run only contract tests (fast, CI-safe)
test-fast:
	pytest tests_contract -v --cov --cov-report=term-missing

# Run everything (contract + heavy)
test-all:
	pytest tests_contract tests_heavy -v

# Run only heavy tests
test-heavy:
	pytest tests_heavy -v -m heavy

# Lint
lint:
	ruff check src/ tests_contract/ tests_heavy/ --fix

# Type check
type-check:
	mypy src/ --no-site-packages

# Format
format:
	black src/ tests_contract/ tests_heavy/
	isort src/ tests_contract/ tests_heavy/
