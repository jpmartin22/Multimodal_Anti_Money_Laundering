# Multimodal Anti Money Laundering

Multimodal AML detection using GNN, DistilBERT, and Bi-LSTM

## Overview

Welcome to Multimodal Anti Money Laundering! This project is designed to provide a scalable, production-ready machine learning pipeline.

## Quick Start

### Installation

```bash
# Using pip
pip install -r requirements.txt

# Using uv (faster alternative)
uv pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Prepare data
make data

# Train the model
make train

# Generate predictions
make predict
```

## Documentation

- [Getting Started](getting_started.md)
- [API Reference](api.md)

## Project Structure

```
multimodal_anti_money_laundering/                  # Repository root
├── src/
│   └── multimodal_anti_money_laundering/          # Importable package (src/ layout)
│       ├── config.py                  # Paths + typed config
│       ├── logging_config.py
│       ├── data/                      # Loaders + raw→processed pipeline
│       ├── features/                  # Feature engineering
│       ├── models/                    # BaseModel ABC + concrete Model
│       ├── evaluation/                # Metric helpers
│       ├── visualization/             # Plot helpers
│       ├── utils/                     # seed, io
│       ├── train_model.py             # Training CLI
│       └── predict_model.py           # Inference CLI
├── data/                              # raw/ and processed/
├── models/                            # Trained artifacts
├── tests/                             # Unit tests
├── docs/                              # MkDocs docs
├── Makefile                           # Common commands
└── pyproject.toml                     # Packaging & deps
```

## License

This project is licensed under the MIT License. See LICENSE for details.
