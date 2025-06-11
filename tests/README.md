# Test Suite Organization

This directory contains the refactored test suite that mirrors the `src/` directory structure for better maintainability and clarity.

## Directory Structure

```
tests/
├── unit/                           # Unit tests mirroring src/ structure
│   ├── data/
│   │   └── test_processing.py      # Tests for src/data/processing.py
│   ├── models/
│   │   ├── test_transformer.py               # Tests for src/models/transformer.py
│   │   ├── test_transformer_generate.py     # Tests for src/models/transformer_generate.py
│   │   └── test_transformer_configurations.py # Model configuration tests
│   ├── training/
│   │   ├── test_manager.py                   # Tests for src/training/manager.py
│   │   ├── test_manager_smart_resumption.py # Smart resumption logic tests
│   │   ├── test_utils.py                     # Tests for src/training/utils.py
│   │   └── test_wandb_integration.py         # W&B integration tests
│   └── utils/
│       ├── test_config_loader.py     # Tests for src/utils/config_loader.py
│       └── test_exceptions.py        # Tests for src/utils/exceptions.py
├── integration/                    # Integration tests
│   ├── training/
│   │   ├── test_training_lifecycle.py        # Training lifecycle integration tests
│   │   ├── test_hyperparameter_sweep.py     # Hyperparameter search tests
│   │   └── test_training_flow.py            # Training flow tests
│   └── generation/
│       └── test_generation_workflows.py     # Generation workflow tests
├── e2e/                           # End-to-end tests for scripts
│   ├── test_train_script.py       # Tests for scripts/train.py
│   └── test_generate_script.py    # Tests for scripts/generate.py & manage_completed_runs.py
├── smoke/                         # Smoke tests
│   ├── test_import_integrity.py   # Import and basic functionality tests
│   └── test_script_functionality.py # Basic script functionality
├── test_data/                     # Test data files
│   └── dummy_data.txt
├── conftest.py                    # Pytest configuration
└── README.md                      # This file
```

## Test Categories

### Unit Tests (`unit/`)

- **Purpose**: Test individual functions and classes in isolation
- **Structure**: Mirrors the `src/` directory structure exactly
- **Example**: `tests/unit/training/test_manager.py` tests `src/training/manager.py`

### Integration Tests (`integration/`)

- **Purpose**: Test how multiple components work together
- **Organization**: Grouped by functional area (training, generation)
- **Example**: `test_training_lifecycle.py` tests the complete training workflow

### End-to-End Tests (`e2e/`)

- **Purpose**: Test complete scripts and workflows from a user perspective
- **Structure**: Mirrors the `scripts/` directory
- **Example**: `test_train_script.py` tests the complete `scripts/train.py` execution

### Smoke Tests (`smoke/`)

- **Purpose**: Quick sanity checks that basic functionality works
- **Use Case**: Run before more comprehensive tests to catch obvious issues

## Key Improvements

### 1. **Clear Module Mapping**

- Easy to find tests for any source file
- `src/training/manager.py` → `tests/unit/training/test_manager.py`

### 2. **Consolidated Related Tests**

- Smart resumption tests are grouped with manager tests
- Model configuration tests are in the models directory
- W&B tests are with training where they're primarily used

### 3. **Logical Test Organization**

- Unit tests focus on individual components
- Integration tests focus on component interactions
- E2E tests focus on user workflows

### 4. **Reduced Redundancy**

- Eliminated duplicate test files
- Consolidated overlapping functionality
- Single source of truth for each test category

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only
pytest tests/e2e/                     # End-to-end tests only
pytest tests/smoke/                   # Smoke tests only

# Run tests for specific modules
pytest tests/unit/training/           # All training unit tests
pytest tests/unit/models/             # All model unit tests

# Run specific test files
pytest tests/unit/training/test_manager.py
pytest tests/integration/training/test_training_lifecycle.py
```

## Test Markers

Tests are marked with the following pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Tests that take significant time
- `@pytest.mark.smoke` - Quick smoke tests

## Migration Notes

### From Old Structure

The old test structure was reorganized as follows:

- `tests/unit/core/` → Split into `tests/unit/data/`, `tests/unit/models/`, `tests/unit/utils/`
- `tests/unit/infrastructure/` → Moved to appropriate directories based on functionality
- `tests/unit/generation/` → Moved to `tests/unit/models/` (generation utilities)
- Complex integration tests → Simplified and consolidated

### Benefits

- **Maintainability**: Easy to find and update tests
- **Clarity**: Clear relationship between source and test files
- **Scalability**: Structure supports growth of the codebase
- **Developer Experience**: Intuitive navigation and test discovery
