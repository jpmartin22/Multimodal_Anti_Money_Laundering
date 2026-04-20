# Getting Started with Multimodal Anti Money Laundering

## Prerequisites

- Python 3.11 or higher
- pip or uv package manager
- Git (for version control)
- Docker (optional, for containerized execution)

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd multimodal_anti_money_laundering
```

### Step 2: Create a Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n multimodal_anti_money_laundering python=3.11
conda activate multimodal_anti_money_laundering
```

### Step 3: Install Dependencies

**Using uv (recommended):**
```bash
pip install uv
uv pip install -r requirements.txt
```

**Using pip:**
```bash
pip install -U pip
pip install -r requirements.txt
```

### Step 4: Set Up Development Environment

```bash
# Install development dependencies
pip install -r requirements_dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

## Running the Project

### Data Processing

Prepare your data for model training:

```bash
make data
```

Or directly:

```bash
python -m multimodal_anti_money_laundering.data.make_dataset
```

### Model Training

Train the machine learning model:

```bash
make train
```

With custom parameters:

```bash
python -m multimodal_anti_money_laundering.train_model --epochs 100 --batch-size 64
```

### Model Prediction

Generate predictions on new data:

```bash
make predict
```

With custom inputs:

```bash
python -m multimodal_anti_money_laundering.predict_model --model-path models/model.pkl --input data/test.csv
```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
pytest tests/ --cov=multimodal_anti_money_laundering

# Run specific test file
pytest tests/test_model.py -v
```

### Code Quality

```bash
# Check for linting issues
make lint

# Auto-format and fix issues
make format

# Type checking
mypy src
```

### Pre-commit Hooks

Pre-commit hooks automatically run before each commit:

```bash
# Manually run pre-commit checks
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

## Docker Usage

### Build Docker Image

```bash
make docker_build
```

Or manually:

```bash
docker build -t multimodal_anti_money_laundering -f dockerfiles/Dockerfile .
```

### Run with Docker Compose

```bash
docker-compose up
```

### Run Single Container

```bash
make docker_run
```

## Project Structure

```
multimodal_anti_money_laundering/                  # Repository root
├── src/
│   └── multimodal_anti_money_laundering/          # Importable package (src/ layout)
│       ├── config.py                  # Paths & typed config
│       ├── logging_config.py
│       ├── data/                      # Loaders + raw→processed pipeline
│       ├── features/                  # Feature engineering
│       ├── models/                    # BaseModel + concrete Model
│       ├── evaluation/                # Metrics
│       ├── visualization/
│       ├── utils/                     # seed, io
│       ├── train_model.py
│       └── predict_model.py
├── tests/                             # Unit tests
├── data/                              # raw/ and processed/
├── models/                            # Trained model artifacts
├── docs/                              # MkDocs documentation
├── configs/                           # Hydra configuration (optional)
├── pyproject.toml
├── requirements.txt
└── Makefile
```

## Configuration

### Using Hydra Configuration
Configuration management is disabled. Modify config in source files as needed.

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, ensure:
1. Virtual environment is activated
2. Dependencies are installed: `pip install -r requirements.txt`
3. Package is installed in editable mode: `pip install -e .`

### CUDA/GPU Issues

If using PyTorch with GPU:

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CPU-only version if needed
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Pre-commit Hook Failures

If pre-commit hooks fail:

```bash
# See what failed
pre-commit run --all-files

# Fix issues manually or with auto-fix
make format

# Try committing again
```

## Next Steps

1. Review the [documentation](index.md)
2. Start with [Phase 1](PHASE1.md) - Data Exploration
3. Check the [API Reference](api.md)

## Support

For issues and questions:
- Check existing [documentation](index.md)
- Review [Phase deliverables](PHASE1.md)
- Contact Anusooya Thimmarayi Neha, Jaya Prakash Yadav Gorla, Preshita Soni, Rajani Meka (nanusooy@depaul.edu, jgorla@depaul.edu, psoni7@depaul.edu, rmeka1@depaul.edu)
